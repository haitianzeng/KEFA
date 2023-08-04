from models.utils import parse_param
from models.speaker_kefa import Speaker as Speaker_kefa


def get_model(args, train_env, listner, tok):
    # Parse model-specific hyper parameters
    hparams = parse_param(args.hparams)

    if args.model_name == 'Speaker_kefa':
        args.hparams = hparams
        return Speaker_kefa(train_env, listner, tok, args)

    return None
