# -*- CODING: UTF-8 -*-
# @time 2023/10/18 21:05
# @Author tyqqj
# @File parser.py
# @
# @Aim This script provides a flexible way to parse command line arguments.
#       The arguments and their default values, types, and help descriptions are loaded from a JSON file.
#       The 'ConfigReader' class reads the JSON file and returns a dictionary of arguments.
#       The 'ArgParser' class takes this dictionary and uses it to create an 'argparse.ArgumentParser' instance.

import argparse
import json
import os

# 参数在以下名称下获取
prm_fname = ["DEFAULT", "param"]


class ConfigReader:
    """Reads a JSON file and returns a dictionary of arguments."""

    def __init__(self, config_file, namecf='None'):
        '''
        只有创建默认的时候才会用到namecf
        :param config_file:
        :param namecf:
        '''
        self.config_file = config_file
        if namecf == 'None':
            try:
                namecf = config_file['name']
            except:
                try:
                    namecf = config_file.split('/')[-1].split('.')[0]
                except:
                    namecf = config_file.split('\\')[-1].split('.')[0]

        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.create_default_config(namecf)

    def create_default_config(self, namecf='None'):
        """Creates a default configuration file if none exists."""
        self.config = {
            'name': namecf,
            'DEFAULT': {
                'param1': {'default': 'default1', 'type': 'str', 'help': 'Description for param1'},
                'param2': {'default': 'default2', 'type': 'str', 'help': 'Description for param2'},
                'param3': {'action': 'store_true', 'help': 'Description for param3'}
            }
        }
        with open(self.config_file, 'w') as configfile:
            json.dump(self.config, configfile, indent=4)

    def get_config(self):
        """Returns the loaded configuration."""
        return self.config


class ArgParser:
    """Takes a dictionary of arguments and uses it to create an 'argparse.ArgumentParser' instance."""

    def __init__(self, config, namecf='uname parser'):
        if namecf == 'None':
            namecf = 'uname parser'

        if namecf == 'uname parser':
            if config['name'] != 'None':
                namecf = config['name']

        self.parser = argparse.ArgumentParser(description=namecf)
        for section, parameters in config.items():
            if section not in prm_fname:
                continue
            for param, param_info in parameters.items():
                help_info = param_info.get('help')
                if 'action' in param_info:
                    self.parser.add_argument(f"--{param}", action=param_info['action'], help=help_info)
                else:
                    default_value = param_info.get('default')
                    type_info = self.str_to_valid_type(param_info.get('type'))
                    self.parser.add_argument(f"--{param}", default=default_value, type=type_info, help=help_info)

    def str_to_valid_type(self, type_str):
        """Converts a type string to a Python type."""
        valid_types = {'int': int, 'float': float, 'str': str, 'bool': bool}
        return valid_types.get(type_str, str)

    def parse_args(self):
        """Parses the command line arguments using the created 'argparse.ArgumentParser' instance."""
        args, _ = self.parser.parse_known_args()
        self.args = args
        return args

    def get_parser(self):
        """Returns the 'argparse.ArgumentParser' instance."""
        return self.parser

    def __str__(self):
        """Returns a string representation of the parsed arguments."""
        if self.args is None:
            return "None"
        else:
            return str(json.dumps(vars(self.args), indent=4))


def parser_to_json(parser, dfname):
    """Converts an argparse.ArgumentParser instance to a JSON-compatible dictionary."""

    parser_dict = {"name": dfname, "DEFAULT": {}}
    # 打印默认生成的参数
    # print([for action in parser._actions if action.dest == "help"])
    # non_help_actions = [action for action in parser._actions if action.dest == "help"]
    # print(non_help_actions)
    for action in parser._actions:
        # Skip the auto-generated help action
        if action.dest == "help":
            # print(action)
            continue
        action_dict = {}

        action_dict['default'] = action.default
        if action.type is not None:
            action_dict['type'] = action.type.__name__
        else:
            action_dict['type'] = None
        action_dict['help'] = action.help
        # Convert type to string representation

        # Only include action for boolean switches
        if type(action).__name__ == '_StoreTrueAction' or type(action).__name__ == '_StoreFalseAction':
            action_dict['action'] = 'store_true' if type(action).__name__ == '_StoreTrueAction' else 'store_false'
        # Remove leading '--' from argument name
        parser_dict["DEFAULT"][action.dest] = action_dict
    return json.dumps(parser_dict, indent=4)


def save_parser_to_json(parser, json_file='./UNETR.json', dfname='None'):
    """Saves an argparse.ArgumentParser instance to a JSON file."""
    if dfname == 'None':
        dfname = json_file.split('/')[-1].split('.')[0]
    parser_dict = parser_to_json(parser, dfname)
    print(parser_dict)
    # 判断是否存在
    if os.path.exists(json_file):
        if input("文件已存在，是否覆盖？(y/n)") == 'y':
            os.remove(json_file)
            print("删除并重新保存")
        else:
            return

    with open(json_file, 'w') as f:
        f.write(parser_dict)


def get_args(config_file, cfname='None', check=True):
    config_reader = ConfigReader(config_file, cfname)
    config = config_reader.get_config()
    arg_parser = ArgParser(config, cfname)

    if check:
        print(arg_parser)

    args = arg_parser.parse_args()
    return args

# Uncomment to use:
# config_reader = ConfigReader('./UNETR.json')
# config = config_reader.get_config()
# arg_parser = ArgParser(config)
# args = arg_parser.parse_args()
#
# print(arg_parser)
# config_reader = ConfigReader('./UNETR.json')
# config = config_reader.get_config()
# arg_parser = ArgParser(config)
# args = arg_parser.parse_args()
#
# file = parser_to_json(arg_parser.get_parser())
# print(file)
# print(arg_parser)
