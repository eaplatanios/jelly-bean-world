/**
 * Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

#ifndef SWIFT_JBW_STATUS_H_
#define SWIFT_JBW_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum JBW_StatusCode {
  JBW_OK = 0,
  JBW_OUT_OF_MEMORY,
  JBW_INVALID_AGENT_ID,
  JBW_VIOLATED_PERMISSIONS,
  JBW_AGENT_ALREADY_ACTED,
  JBW_AGENT_ALREADY_EXISTS,
  JBW_SERVER_PARSE_MESSAGE_ERROR,
  JBW_CLIENT_PARSE_MESSAGE_ERROR,
  JBW_SERVER_OUT_OF_MEMORY,
  JBW_CLIENT_OUT_OF_MEMORY,
  JBW_IO_ERROR,
  JBW_LOST_CONNECTION,
  JBW_INVALID_SIMULATOR_CONFIGURATION,
  JBW_MPI_ERROR
} JBW_StatusCode;

// Represents a Jelly Bean World (JBW) API call status.
// This is a struct to allow providing more information (e.g., error messages) in the future.
typedef struct JBW_Status {
  JBW_StatusCode code;
} JBW_Status;

// Returns a new status object.
JBW_Status* JBW_NewStatus();

// Deletes a previously created status object.
void JBW_DeleteStatus(JBW_Status* status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SWIFT_JBW_STATUS_H_ */
