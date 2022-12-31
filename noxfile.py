from nox_poetry import Session, session

python_versions = ["3.9", "3.10", "3.11"]


@session(python=python_versions)  # type: ignore[misc]
def tests(session: Session) -> None:
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", "--no-dev", external=True)
    session.install("coverage[toml]", "pytest", "pytest-cov", "pytest-mock")
    session.run("pytest", *args)


locations = "stratified_models", "noxfile.py"


@session(python="3.11")  # type: ignore[misc]
def lint(session: Session) -> None:
    args = session.posargs or locations
    session.install("flake8", "flake8-import-order")
    session.run("flake8", *args)


@session(python="3.11")  # type: ignore[misc]
def black(session: Session) -> None:
    args = session.posargs or locations
    session.install("black")
    session.run("black", "--diff", "--color", *args)


@session(python=python_versions)  # type: ignore[misc]
def mypy(session: Session) -> None:
    args = session.posargs or locations
    session.install("mypy")
    session.run("mypy", *args)


package = "stratified_models"


@session(python=python_versions)  # type: ignore[misc]
def typeguard(session: Session) -> None:
    args = session.posargs or []
    session.run("poetry", "install", "--no-dev", external=True)
    session.install("pytest", "typeguard")
    session.run("pytest", f"--typeguard-packages={package}", *args)
