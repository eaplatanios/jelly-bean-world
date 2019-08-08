# Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from __future__ import absolute_import, division, print_function

__all__ = ['Permissions', 'GRANT_ALL_PERMISSIONS', 'DENY_ALL_PERMISSIONS']

class Permissions:
  """Permissions that can control/limit the behavior of MPI clients."""

  add_agent = False
  remove_agent = False
  remove_client = False
  set_active = False
  get_map = False
  get_agent_ids = False

  def __repr__(self):
    return repr((self.add_agent, self.remove_agent, self.remove_client, self.set_active, self.get_map, self.get_agent_ids))

  def __str__(self):
    return str((self.add_agent, self.remove_agent, self.remove_client, self.set_active, self.get_map, self.get_agent_ids))

GRANT_ALL_PERMISSIONS = Permissions()
GRANT_ALL_PERMISSIONS.add_agent = True
GRANT_ALL_PERMISSIONS.remove_agent = True
GRANT_ALL_PERMISSIONS.remove_client = True
GRANT_ALL_PERMISSIONS.set_active = True
GRANT_ALL_PERMISSIONS.get_map = True
GRANT_ALL_PERMISSIONS.get_agent_ids = True

DENY_ALL_PERMISSIONS = Permissions()
DENY_ALL_PERMISSIONS.add_agent = False
DENY_ALL_PERMISSIONS.remove_agent = False
DENY_ALL_PERMISSIONS.remove_client = False
DENY_ALL_PERMISSIONS.set_active = False
DENY_ALL_PERMISSIONS.get_map = False
DENY_ALL_PERMISSIONS.get_agent_ids = False
