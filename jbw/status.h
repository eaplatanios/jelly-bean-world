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

namespace jbw {

typedef uint8_t status_type;

enum class status : uint8_t {
  OK = 0,
  OUT_OF_MEMORY,
  INVALID_AGENT_ID,
  PERMISSION_ERROR,
  AGENT_ALREADY_ACTED,
  AGENT_ALREADY_EXISTS,
  SERVER_PARSE_MESSAGE_ERROR,
  CLIENT_PARSE_MESSAGE_ERROR,
  SERVER_OUT_OF_MEMORY,
  CLIENT_OUT_OF_MEMORY
};

/**
 * Reads a status from `in` and stores the result in `s`.
 */
template<typename Stream>
inline bool read(status& s, Stream& in) {
	status_type v;
	if (!read(v, in)) return false;
	s = (status) v;
	return true;
}

/**
 * Writes the given status `s` to the stream `out`.
 */
template<typename Stream>
inline bool write(const status& s, Stream& out) {
	return write((status_type) s, out);
}

} /* namespace jbw */

#endif /* JBW_STATUS_H_ */
