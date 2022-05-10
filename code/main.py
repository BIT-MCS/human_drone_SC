from util import *

os.environ['MKL_NUM_THREADS'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str, help='the name of environment (KAIST or NCSU)')
    parser.add_argument('method_name', type=str, help='the name of method (fd_mappo_cubicmap)')
    parser.add_argument('mode', type=str, help='train or test')
    args = parser.parse_args()

    ENV_CONF = importlib.import_module('environment.' + args.env_name + '.conf').ENV_CONF
    Env = importlib.import_module('environment.' + args.env_name + '.env').Env
    importlib.import_module('method.' + args.method_name + '.' + args.mode).main(ENV_CONF, Env)
