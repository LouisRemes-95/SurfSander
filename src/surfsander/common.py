from rich.console import Console


console = Console(stderr=True)


class UserError(Exception):
    pass