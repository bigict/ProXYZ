import click

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from proxyz.models.processing_xyz import XYZProcessor
from proxyz.data import dataset

XYZProcessor.register_for_auto_class()

class dict2object(object):
    def __init__(self, **args):
        self.__dict__.update(args)

@click.command(context_settings={'show_default': True})
@click.option(
    "--output_dir",
    type=click.Path(),
    default="./deepseek_style_model",
    help="Where checkpoints and the final model are saved.",
)
@click.option(
    "--tokenizer_file",
    type=click.Path(),
    default="my_tokenizer.json",
    help="Path to the tokenizer json file.",
)
def main(**args):
    args = dict2object(**args)

    fim_tokens = dataset.FIM_TOKENS

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_file,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    processor = XYZProcessor(tokenizer=tokenizer)
    print(tokenizer.all_special_tokens, tokenizer.all_special_ids)
    print(processor.attributes)
    print(dir(processor))
    print(processor(["ABC", "ACCGGGWGCG"], fim_apply=False, char_apply=True))

    processor.save_pretrained(args.output_dir)

    processor = AutoProcessor.from_pretrained(args.output_dir, trust_remote_code=True)
    print(processor.attributes)
    print(dir(processor))


    print(processor(["ABC", "ACCGGGWGCG"], fim_apply=False, char_apply=True))

if __name__ == "__main__":
    main()

