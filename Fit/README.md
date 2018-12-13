# Fazer fit de CDF

Extraia dos seus dados a coluna que contem os dados da CDF analisada
e crie um arquivo onde cada linha contenha o referido dado
```
 cat dados.csv | awk -F "," '{print $2}' | sed "1d" > dados_to_fit.csv
 ```

Chama o script q testa as distribuições

```
python distribution-check.py -f dados/dados_to_fit.csv -p -n 4 -d
```
