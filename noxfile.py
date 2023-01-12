from nox_poetry import Session, session

latest_python_version = "3.11"
old_python_versions = ["3.9", "3.10"]


@session(python=latest_python_version)  # type: ignore[misc]
def tests_with_coverage(session: Session) -> None:
    args = session.posargs or ["--cov"]
    session.run("pip", "install", "cvxpy~=1.3.0")
    session.run("poetry", "install", "--no-dev", external=True)
    session.install("coverage[toml]", "pytest", "pytest-cov", "pytest-mock")
    session.run("pytest", *args)


@session(python=old_python_versions)  # type: ignore[misc]
def tests(session: Session) -> None:
    args = session.posargs
    session.run("pip", "install", "cvxpy~=1.3.0")
    session.run("poetry", "install", "--no-dev", external=True)
    session.install("pytest", "pytest-mock")
    session.run("pytest", *args)


locations = "stratified_models", "noxfile.py"


@session(python=latest_python_version)  # type: ignore[misc]
def lint(session: Session) -> None:
    args = session.posargs or locations
    session.install("flake8")
    session.run("flake8", *args)


@session(python=latest_python_version)  # type: ignore[misc]
def black(session: Session) -> None:
    args = session.posargs or locations
    session.install("black")
    session.run("black", "--diff", "--color", *args)


@session(python=latest_python_version)  # type: ignore[misc]
def mypy(session: Session) -> None:
    args = session.posargs or locations
    session.run("pip", "install", "cvxpy~=1.3.0")
    session.run("poetry", "install", "--no-dev", external=True)
    session.install("mypy", "pytest")
    session.run("mypy", "--version")
    session.run("mypy", *args)


package = "stratified_models"


@session(python=latest_python_version)  # type: ignore[misc]
def typeguard(session: Session) -> None:
    args = session.posargs or []
    session.run("pip", "install", "cvxpy~=1.3.0")
    session.run("poetry", "install", "--no-dev", external=True)
    session.install("pytest", "typeguard")
    session.run("pytest", f"--typeguard-packages={package}", *args)
