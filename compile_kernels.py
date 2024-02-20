import argparse

from megatron.fused_kernels import load


def main():
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)
    args = parser.parse_args()
    args.masked_softmax_fusion = True
    args.gradient_accumulation_fusion = True
    args.rank = 0

    load(args)


if __name__ == '__main__':
    main()
