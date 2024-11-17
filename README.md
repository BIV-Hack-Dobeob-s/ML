# ML

# Запуск

Перед запуском создайте в директории `input/` файл `payments.tsv` и поместите в него ваши данные.

После этого из корня проекта выполните следующую команду:

```sh
docker build -t dobeobs . && docker run --rm --name dobeobs-instance -v "./output:/app/output" -v "./input:/app/input" dobeobs
```

После выполнения команды в директории `output/` создастся файл `output.csv`, содержащий выходные данные.
