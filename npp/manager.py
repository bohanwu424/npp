from aim import Run
import os
import randomname


def create_run(subproject, args):
    """Set up an AIM run, give it a human-readable name and a local directory."""
    run = Run(log_system_params=True, repo=subproject)
    run_name = randomname.get_name()
    local_path = os.path.join(subproject, run_name + '-' + run.hash)
    run['local_dir'] = local_path
    os.makedirs(local_path)
    run['args'] = args.__dict__
    run.add_tag(run_name)

    return run