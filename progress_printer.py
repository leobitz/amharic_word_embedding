from check_point_manager import CheckpointManager
import time
import argparse

def parse_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', dest='name', type=str)
    parser.add_argument('-t' '--model_tag', dest='tag', type=str)
    parser.add_argument('-d', '--duration', dest='duration', default=60, type=int)
    args = parser.parse_args()
    return args

args = parse_training_args()

duration = args.duration
name = "{0}_{1}".format(args.name, args.tag)
checkpoint = CheckpointManager(name)
while True:
    state = checkpoint.get_last_state()
    if state is None:
        print('No state')
    else:
        print(state[1])
    time.sleep(duration)

    





