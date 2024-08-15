import argparse
import os

import torch
import torchutils
import torchvision.transforms.v2 as T
from sklearn.metrics import classification_report
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./data')
    parser.add_argument('--output', default='./checkpoint')
    parser.add_argument('--strategy', default='ddp')
    parser.add_argument('--compile', default='none', type=str)
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--num-workers', default=4)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


def main():
    args = get_args()

    if args.compile == 'none':
        args.compile = False

    torchutils.set_seeds(args.seed)

    torchutils.makedirs0(args.output, exist_ok=True)
    logger = torchutils.get_logger(
        f'[rank {torchutils.get_rank()}] CIFAR10',
        filename=os.path.join(args.output, 'log.log') if torchutils.is_primary() else None,
    )
    device = torchutils.get_device()

    # First download.
    if torchutils.is_primary():
        CIFAR10(args.dataset, download=True)
    # we must sync here.
    torchutils.wait_for_everyone()

    # Datasets
    train = CIFAR10(
        args.dataset,
        train=True,
        transform=T.Compose(
            [
                T.ToImage(),
                T.ToDtype(dtype=torch.float32, scale=True),
                T.Resize((32, 32)),
                T.RandomHorizontalFlip(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
    )
    test = CIFAR10(
        args.dataset,
        train=False,
        transform=T.Compose(
            [
                T.ToImage(),
                T.ToDtype(dtype=torch.float32, scale=True),
                T.Resize((32, 32)),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
    )
    repr_kwargs = torchutils.get_dataloader_kwargs()
    trainloader = torchutils.create_dataloader(
        train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, **repr_kwargs
    )
    testloader = torchutils.create_dataloader(test, batch_size=args.batch_size, shuffle=False)

    # Model
    model = resnet18(num_classes=len(train.class_to_idx))
    model.to(device)
    wrapped, compiled = torchutils.wrap_module(model, args.strategy, args.compile)

    # Optimizer
    optimizer = torch.optim.Adam(compiled.parameters(), lr=args.lr)

    # Scheduler
    scheduler = torchutils.create_scheduler(optimizer, 'multistep', args.epochs, milestones=[60, 80], gamma=0.1)

    # Criterion
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # AMP
    grad_scaler = torchutils.get_grad_scaler(args.amp, is_fsdp=args.strategy == 'fsdp')

    # vars.
    epoch = 0
    best_loss = 999

    # load checkpoint.
    ckpt_folder = os.path.join(args.output, 'bins')
    consts = torchutils.load_checkpoint(
        ckpt_folder,
        allow_empty=True,
        # These must be the same structure as input of save_checkpoint.
        model=compiled,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=grad_scaler,
        others={'best_loss': best_loss, 'epoch': epoch},
    )
    epoch = consts.get('epoch', epoch)
    best_loss = consts.get('best_loss', best_loss)

    logger.info(f'Start epoch {epoch}, current best loss: {best_loss}')

    while epoch < args.epochs:
        if hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch)

        # Main loop
        model.train()
        loss, correct, total = 0, 0, 0
        for batch in trainloader:
            image = batch[0].to(device)
            label = batch[1].to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                output = compiled(image)
                batch_loss = criterion(output, label)

            grad_scaler.scale(batch_loss).backward()
            grad_scaler.step(optimizer)
            optimizer.zero_grad()
            grad_scaler.update()

            batch_size = image.size(0)
            total += batch_size
            loss += batch_loss.item() * batch_size
            predicted = torch.max(output.data, dim=1)[1]
            correct += (predicted == label).sum().item()

        scheduler.step()
        epoch += 1

        # Logging
        total = torchutils.reduce(torch.tensor([total], device=device))
        loss = torchutils.reduce(torch.tensor([loss], device=device)) / total
        accuracy = torchutils.reduce(torch.tensor([correct], device=device)) / (total) * 100

        logger.info(
            ' | '.join([f'{epoch} / {args.epochs}', f'BCE: {loss.item(): 7.3f}', f'Accuracy: {accuracy.item():.2f}%'])
        )

        # Save best model. (This should usually be done on validation loss.)
        loss = loss.item()
        if loss < best_loss:
            best_loss = loss
            torchutils.save_model(args.output, model=compiled, name='best-model')

        # Checkpointing.
        torchutils.save_checkpoint(
            ckpt_folder,
            model=compiled,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
            others={'best_loss': best_loss, 'epoch': epoch},
        )

    # Load best.
    torchutils.load_model(args.output, compiled, name='best-model', map_location=device)
    compiled.eval()

    # Test
    with torch.no_grad() if args.compile else torch.inference_mode():
        loss, correct, total, predictions, ground_truths = 0, 0, 0, [], []
        for batch in testloader:
            image = batch[0].to(device)
            label = batch[1].to(device)

            output = compiled(image)
            batch_loss = criterion(output, label)

            batch_size = image.size(0)
            total += batch_size
            loss += batch_loss.item() * batch_size
            predicted = torch.max(output.data, dim=1)[1]
            correct += (predicted == label).sum().item()

            predictions.append(predicted)
            ground_truths.append(label)

    predictions = torchutils.gather(torch.cat(predictions), dst=0)
    ground_truths = torchutils.gather(torch.cat(ground_truths), dst=0)

    if torchutils.is_primary():
        logger.info(
            '\n' + \
            classification_report(
                ground_truths.cpu().numpy(),
                predictions.cpu().numpy(),
                target_names=train.classes,
                digits=5,
            )
        )

    torchutils.destroy_process_group()


if __name__ == '__main__':
    main()
