"""Read saved agent metadata without enabling arbitrary Python YAML objects."""
from pathlib import Path
import yaml


class AgentConfigLoader(yaml.SafeLoader):
    pass


AgentConfigLoader.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    lambda loader, node: loader.construct_sequence(node),
)


def load_agent_config(path):
    with Path(path).open(encoding='utf-8') as stream:
        return yaml.load(stream, Loader=AgentConfigLoader)
