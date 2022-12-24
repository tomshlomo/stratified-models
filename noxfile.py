from nox_poetry import session


python_versions = ["3.9", "3.10", "3.11"]


@session(python=python_versions)
def tests(session):
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", "--no-dev", external=True)
    session.install("coverage[toml]", "pytest", "pytest-cov", "pytest-mock")
    session.run("pytest", *args)


locations = "stratified_models", "noxfile.py"


@session(python="3.11")
def lint(session):
    args = session.posargs or locations
    session.install("flake8", "flake8-import-order")
    session.run("flake8", *args)


@session(python="3.11")
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", "--diff", "--color", *args)
