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

#ifndef JBW_STATUS_H_
#define JBW_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

// Represents a Jelly Bean World (JBW) error code.
typedef enum JBW_ErrorCode : unsigned char {
  JBW_OK = 0,
  JBW_UNKNOWN_ERROR,
  JBW_OUT_OF_MEMORY_ERROR,
  JBW_IO_ERROR,
  JBW_COMMUNICATION_ERROR,
  JBW_INVALID_SIMULATOR_CONFIGURATION,
  JBW_SERVER_INITIALIZATION_FAILURE,
  JBW_LOST_CONNECTION,
  JBW_EXCEEDED_AGENT_LIMIT,
} JBW_ErrorCode;

// Represents a Jelly Bean World (JBW) API call status.
typedef struct JBW_Status {
  JBW_ErrorCode code;
  const char* message;
} JBW_Status;

// Returns a new status object.
JBW_Status* JBW_NewStatus();

// Deletes a previously created status object.
void JBW_DeleteStatus(JBW_Status* status);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* JBW_STATUS_H_ */
