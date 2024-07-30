# torchutils

Copy & paste-able utilities for writing training codes with PyTorch in a single file.

Feel free to modify.

## Notes on distributed training

- Launch training codes using the `torchrun` command.

- This file is NOT tested on multi-node environments and is likely to fail.

## Examples

- Image classification: [train_cifar10_resnet18.py](./train_cifar10_resnet18.py)

    Single GPU + AMP + torch.compile().

    ```sh
    python train_cifar10_resnet18.py --amp --compile=default
    ```

    For multi GPU, you only need to replace the `python` command with the `torchrun` command.

    ```sh
    torchrun --nproc-per-node=2 train_cifar10_resnet18.py --amp --compile=default
    ```

## License

[MIT License](./LICENSE)

## Author(s)

Tomoya Sawada
