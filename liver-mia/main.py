import argparse
from data import LiverDataset
from utils_train import get_victim, get_shadow, get_attack
from global_attack import global_attack 


def main(args):
    # read all the paths from the liver folder
    data = LiverDataset(args.trainsize)

    # train victim and shadow models
    print(" ** TRAINING VICTIM MODEL **")
    victim_model, _ = get_victim(data, args)
    print(" ** TRAINING SHADOW MODEL **")
    shadow_model, shadow_threshold = get_shadow(data, args)

    print(" ** MEMBERSHIP INFERENCE ATTACK **")
    if args.attacktype == 3:
        global_attack(data, args, victim_model, shadow_threshold)
    else:
        _ = get_attack(data, args, victim_model, shadow_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # resnets, mobilenet_v2, vgg11
    parser.add_argument("--victim", type=str, default='resnet34') 
    # resnets, mobilenet_v2, vgg11
    parser.add_argument("--shadow", type=str, default='resnet34')
    # 1 -- Type-I attack, 
    # 2 -- Type-II attack,
    # 3 -- Global loss-based attack
    parser.add_argument("--attacktype", type=int, default=2)
    # 1 -- no defense,
    # 2 -- argmax defense,
    # 3 -- crop training,
    # 4 -- mix-up,
    # 5 -- min-max,
    # 6 -- DP
    parser.add_argument("--defensetype", type=int, default=0)
    # same for victim and shadow models, [500 - 2000]
    parser.add_argument("--trainsize", type=int, default=500)
    args = parser.parse_args()

    print("Victim encoder: {}".format(args.victim))
    print("Shadow encoder: {}".format(args.shadow))
    attacks = ['Type-I', 'Type-II', 'Global-loss']
    print("Attack type: {}".format(attacks[args.attacktype - 1]))
    defenses = ['No defense', 'Argmax', 'Crop-training', 'Mix-up', 'Min-max', 'DP']
    print("Defense type: {}".format(defenses[args.defensetype - 1]))
    print("Train size: {}".format(args.trainsize))
    print()

    main(args)