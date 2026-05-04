import json
import requests
import warnings
import argparse
import os

DEFAULT_HOST = 'https://labelstudio.brick.whoi.edu'


def load_config(path):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


def resolve_config(args):
    config = load_config(args.config)

    host = args.host or config.get("host") or DEFAULT_HOST
    token = args.token or config.get("token")
    project_id = args.project_id or config.get("project_id")

    if token is None:
        raise ValueError("TOKEN must be provided via CLI or config file")
    if project_id is None:
        raise ValueError("PROJECT_ID must be provided via CLI or config file")

    return host, token, project_id


def get_task(host: str,
             token: str,
             project_id: int,
             media_name: str,
             frame_number: int,
             resolve_uri: bool = False):

    url = f"{host}/api/tasks"

    payload = dict(project=project_id)

    # Response Format #
    payload['include'] = ['id', 'data']
    if not resolve_uri:
        payload['resolve_uri'] = resolve_uri

    # Filtering #
    filter_dict = {
        "conjunction": "and",
        "items": [
            {
                "filter": "filter:tasks:data.media",
                "operator": "equal",
                "value": media_name,
                "type": "String"
            },
            {
                "filter": "filter:tasks:data.frame",
                "operator": "equal",
                "value": frame_number,
                "type": "Number"
            }
        ]
    }

    payload['query'] = json.dumps({"filters": filter_dict})

    headers = {"Authorization": f"Token {token}"}
    response = requests.get(url, params=payload, headers=headers)

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ValueError(f"{e} | {detail}") from e

    tasks = response.json().get('tasks', [])

    if len(tasks) > 1:
        warnings.warn("Multiple Tasks Matched!")
        return tasks
    elif len(tasks) == 0:
        warnings.warn('Task Not Found')
        return None
    else:
        return tasks[0]


def create_link(host: str, project_id: int, task_id: int):
    return f'{host}/projects/{project_id}/data?task={task_id}'


def main():
    parser = argparse.ArgumentParser(description="Query Label Studio task")

    parser.add_argument("-m", "--media", required=True, help="Media name")
    parser.add_argument("-f", "--frame", required=True, type=int, help="Frame number")

    parser.add_argument("--host", help="Label Studio host URL")
    parser.add_argument("--token", help="API token")
    parser.add_argument("--project_id", type=int, help="Project ID")

    parser.add_argument("-c", "--config", help="Path to config JSON")

    args = parser.parse_args()

    host, token, project_id = resolve_config(args)

    task = get_task(
        host=host, token=token,
        project_id=project_id,
        media_name=args.media,
        frame_number=args.frame,
    )
    
    if task and isinstance(task,dict):
        target_url = create_link(host, project_id, task['id'])
        task['labelstudio_task_url'] = target_url
    
    #print(task)  # to see the task-data and id
    print(target_url)



if __name__ == "__main__":
    main()
