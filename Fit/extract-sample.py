#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
from argparse import RawTextHelpFormatter
import pandas as pd


parser = argparse.ArgumentParser(
    description='''\033[31m \033[1m \n
╔╦╗┌─┐┬┌─┌─┐  ╔═╗┌─┐┌┬┐┌─┐┬  ┌─┐  ╔═╗╔╦╗╔═╗
║║║├─┤├┴┐├┤   ╚═╗├─┤│││├─┘│  ├┤   ║   ║║╠╣
╩ ╩┴ ┴┴ ┴└─┘  ╚═╝┴ ┴┴ ┴┴  ┴─┘└─┘  ╚═╝═╩╝╚
     \n Cria uma amostra mantendo o shape da CDF original \033[0m \033[0m ''',
    epilog="""Be Free!""", formatter_class=RawTextHelpFormatter)
parser.add_argument('--file', '-f', required=True, help='Arquivos com os dados')
parser.add_argument('--column', '-c', required=True, type=int, help='Numero da coluna, iniciando no 0')
parser.add_argument('--divisor', '-d', default=1000.0, type=float,
                    help='Divide a quantidade de dados da amostra pelo parametro (Default: 1000)')
args = parser.parse_args()

print(args)

dir_arquivo_in = os.path.abspath(os.path.dirname(args.file))

df = pd.read_csv(args.file)
df.head()
df.columns[args.column]

x = df.sort_values(by=df.columns[args.column])[df.columns[args.column]].values

with open("{}/sample_{}".format(dir_arquivo_in, os.path.basename(args.file)), "a") as f:
    for idx, val in enumerate(x):
        if ((idx % args.divisor) == 0):
            f.write(str(val)+"\n")
