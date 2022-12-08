import json
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeRemainingColumn
from lognormal_errors import params_solve

data = json.loads(open('autism-trans.json', 'r').read())

results = {}

pbar = Progress(
    SpinnerColumn(style="bold magenta"),
    "[progress.description]{task.description}",
    BarColumn(complete_style="bold green", finished_style="bold blue"),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeRemainingColumn(),
    )

with pbar:
    categories = pbar.add_task('Processing categories ...', total=len(data), style='Bold green')
    for category in data:
        pbar.update(categories, advance=1)
        datasets = pbar.add_task(
            f"> 📁⏳ Processing category {category} ({len(data[category])} datasets) ...",
            style='Bold purple', total=len(data[category])
            )
        swap = False if category != "trans given autism" else True
        results[category] = {}
        for dataset in data[category]:
            OR = data[category][dataset]['OR']
            CI = data[category][dataset]['95%CI']
            solve = pbar.add_task(f">> min → ƒ() Now finding parameters for {dataset} 🤓", total=1, style='Bold blue')
            results[category][dataset] = params_solve(CI, logit=False, maxiter=5000, display=False, swap=swap)
            pbar.update(solve, advance=100, display=False)
            pbar.update(datasets, advance=1)

with open('autism-probas.json', 'w+') as f:
    json.dump(results, f, indent=4)