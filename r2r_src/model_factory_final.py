from models_final.utils import parse_param
from models_final.speaker_ori import Speaker as Speaker_ori
from models_final.speaker_subins_dtw_new5 import Speaker as Speaker_subins_dtw_new5
from models_final.speaker_final_1 import Speaker as Speaker_final_1


def get_model(args, train_env, listner, tok):
    # Parse model-specific hyper parameters
    hparams = parse_param(args.hparams)

    if args.model_name == 'speaker_ori':
        return Speaker_ori(train_env, listner, tok, args)

    if args.model_name == 'speaker_subins_dtw_new5':
        args.hparams = hparams
        return Speaker_subins_dtw_new5(train_env, listner, tok, args)

    if args.model_name == 'speaker_final_1':
        args.hparams = hparams
        return Speaker_final_1(train_env, listner, tok, args)

    return None
