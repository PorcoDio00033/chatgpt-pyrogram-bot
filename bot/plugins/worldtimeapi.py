import os
from datetime import datetime
from typing import Dict, List
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .plugin import Plugin


class WorldTimeApiPlugin(Plugin):
    """
    A plugin to get the current time from a given timezone, using local system data.
    """

    def __init__(self):
        default_timezone = os.getenv('WORLDTIME_DEFAULT_TIMEZONE')
        if not default_timezone:
            raise ValueError('WORLDTIME_DEFAULT_TIMEZONE environment variable must be set to use WorldTimeApiPlugin')
        self.default_timezone = default_timezone

    def get_source_name(self) -> str:
        return 'LocalZoneInfo'

    def get_spec(self) -> List[Dict]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'worldtimeapi',
                    'description': 'Get the current time and date from a given timezone',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'timezone': {
                                'type': 'string',
                                'description': 'The timezone identifier (e.g: `Europe/Rome`). Infer this from the location.'
                                f'Use {self.default_timezone} if not specified.',
                            }
                        },
                        'required': ['timezone'],
                        'additionalProperties': False,
                    },
                    'strict': True,
                },
            }
        ]

    async def execute(self, function_name, helper, **kwargs) -> Dict:
        timezone_str = kwargs.get('timezone', self.default_timezone)
        
        try:
            tz = ZoneInfo(timezone_str)
        except ZoneInfoNotFoundError:
            return {'error': f"Timezone '{timezone_str}' not found."}

        # Get current time in that timezone
        wtr_obj = datetime.now(tz)
        
        time_24hr = wtr_obj.strftime('%H:%M:%S')
        time_12hr = wtr_obj.strftime('%I:%M:%S %p')
        date_str = wtr_obj.strftime('%Y-%m-%d')

        return {'24hr': time_24hr, '12hr': time_12hr, 'date': date_str, 'timezone': timezone_str}
