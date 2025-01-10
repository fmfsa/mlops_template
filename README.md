# mlops_template

Template for mlops

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── preprocess_data.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_preprocess_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Notes
To save new dependencies, use the following command:
Either use pipreqs or pip freeze (not recommended):
```bash
pipreqs .
```

For format and linting, use the following commands:
```bash
ruff check .
ruff format .
```

To run locally in dev, use the following command:
```bash
pip install -e .
train
evaluate
visualize
```

To create and run docker images
```bash
docker build -f dockerfiles/train.dockerfile . -t train:latest
docker run --name train --rm -v $(pwd)/models/model.pth:/models/model.pth -v $(pwd)/data/test_images.pt:/data/test_images.pt -v $(pwd)/data/test_targets.pt:/data/test_targets.pt train:latest

docker build -f dockerfiles/evaluate.dockerfile . -t evaluate:latest
docker run --name evaluate --rm -v $(pwd)/models/model.pth:/models/model.pth -v $(pwd)/data/test_images.pt:/data/test_images.pt -v $(pwd)/data/test_targets.pt:/data/test_targets.pt evaluate:latest ../models/model.pth
```

