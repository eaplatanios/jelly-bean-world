/**
 * json.h - Simple JSON parser.
 *
 *  Created on: Feb 27, 2017
 *      Author: asaparov
 */

#ifndef JSON_H_
#define JSON_H_

#include <core/array.h>
#include <core/lex.h>
#include <ctype.h>

using namespace core;

enum json_state {
	JSON_KEY,
	JSON_VALUE,
	JSON_FIRST_KEY,
	JSON_FIRST_VALUE,
	JSON_COLON,
	JSON_COMMA,
	JSON_STRING,
	JSON_KEYWORD,
	JSON_NUMBER,
	JSON_END
};

enum json_context {
	JSON_CONTEXT_KEY,
	JSON_CONTEXT_VALUE,
	JSON_CONTEXT_LIST
};

template<typename Reader>
bool emit_keyword(const array<char>& token, const position& pos, Reader& reader) {
	if (compare_strings(token, "true")) {
		return emit_true(pos, reader);
	} else if (compare_strings(token, "false")) {
		return emit_false(pos, reader);
	} else if (compare_strings(token, "null")) {
		return emit_null(pos, reader);
	} else {
		read_error("Unrecognized keyword", pos);
		return false;
	}
}

template<typename Reader>
bool emit_number(const array<char>& token, const position& pos, Reader& reader) {
	double value;
	if (!parse_float(token, value)) {
		read_error("Unable to interpret numerical value", pos);
		return false;
	}
	return emit_number(value, pos, reader);
}

inline bool codepoint_to_utf8(array<char>& str, unsigned int codepoint)
{
	if (!str.ensure_capacity(str.length + 3)) return false;
	if (codepoint <= 0x7F) {
		str[str.length] = codepoint;
		str.length++;
	} else if (codepoint <= 0x7FF) {
		str[str.length] = 0xC0 | (codepoint >> 6);
		str[str.length + 1] = 0x80 | (codepoint & 0x3F);
		str.length += 2;
	} else if (codepoint <= 0xFFFF) {
		str[str.length] = 0xE0 | (codepoint >> 12);
		str[str.length + 1] = 0x80 | ((codepoint >> 6) & 0x3F);
		str[str.length + 2] = 0x80 | (codepoint & 0x3F);
		str.length += 3;
	} else if (codepoint <= 0x10FFFF) {
		fprintf(stderr, "codepoint_to_utf8 ERROR: Codepoints larger than 0xFFFF are unsupported.\n");
		return false;
		/*str[str.length] = 0xF0 | (codepoint >> 18);
		str[str.length + 1] = 0x80 | ((codepoint >> 12) & 0x3F);
		str[str.length + 2] = 0x80 | ((codepoint >> 6) & 0x3F);
		str[str.length + 3] = 0x80 | (codepoint & 0x3F);
		str.length += 4;*/
	}
	return true;
}

template<typename Stream>
bool emit_escape(Stream& in, array<char>& token, int& next, position& current) {
	if (next == -1) {
		read_error("Unexpected end of input", current);
		return false;
	} else if (next == 'n') {
		if (!token.add('\n')) return false;
	} else if (next == 'b') {
		if (!token.add('\b')) return false;
	} else if (next == 'f') {
		if (!token.add('\f')) return false;
	} else if (next == 't') {
		if (!token.add('\t')) return false;
	} else if (next == 'r') {
		if (!token.add('\r')) return false;
	} else if (next == 'u') {
		/* read the next four characters */
		char escape[4];
		for (unsigned int i = 0; i < 4; i++) {
			next = fgetc(in);
			current.column++;
			if (next == -1) {
				read_error("Unexpected end of input", current);
				return false;
			}
			escape[i] = (char) next;
		}
		unsigned int codepoint;
		if (!parse_uint(escape, codepoint, 16) || !codepoint_to_utf8(token, codepoint)) {
			read_error("Unable to interpret Unicode codepoint", current);
			return false;
		}
	} else if (!token.add(next)) return false;
	return true;
}

template<typename Stream, typename Reader>
bool json_parse(Stream& in, Reader& reader)
{
	position current = position(1, 1);
	array<char> token = array<char>(1024);
	json_state state = JSON_VALUE;

	int next = fgetc(in);
	bool new_line = false;
	array<json_context> contexts = array<json_context>(8);
	while (next != -1) {
		switch (state) {
		case JSON_KEY:
		case JSON_FIRST_KEY:
			if (next == '"') {
				state = JSON_STRING;
			} else if (next == '}') {
				if (state != JSON_FIRST_KEY || contexts.length == 0 || contexts.last() != JSON_CONTEXT_VALUE) {
					read_error("Unexpected closing brace '}'", current);
					return false;
				}
				end_object(current, reader);
				contexts.pop();
				if (contexts.length == 0) {
					state = JSON_END;
				} else { /* we don't check the previous context since the FSM doesn't allow JSON_CONTEXT_KEY */
					state = JSON_COMMA;
				}
			} else if (next == ' ' || next == '\t' || next == '\n' || next == '\r') {
				new_line = (next == '\n');
			} else {
				read_error("Expected a key-value pair or closing brace '}' for object", current);
				return false;
			}
			break;

		case JSON_VALUE:
		case JSON_FIRST_VALUE:
			if (next == '{') {
				state = JSON_FIRST_KEY;
				if (!contexts.add(JSON_CONTEXT_KEY)) return false;
				begin_object(current, reader);
			} else if (next == '[') {
				state = JSON_FIRST_VALUE;
				if (!contexts.add(JSON_CONTEXT_LIST)) return false;
				begin_list(current, reader);
			} else if (next == '"') {
				state = JSON_STRING;
			} else if (isdigit(next) || next == '+' || next == '-' || next == '.') {
				if (!token.add(next)) return false;
				state = JSON_NUMBER;
			} else if (next == ' ' || next == '\t' || next == '\n' || next == '\r') {
				new_line = (next == '\n');
			} else if (next == ']') {
				if (state != JSON_FIRST_VALUE || contexts.length == 0 || contexts.last() != JSON_CONTEXT_LIST) {
					read_error("Unexpected closing bracket ']'", current);
					return false;
				}
				end_list(current, reader);
				contexts.pop();
				if (contexts.length == 0) {
					state = JSON_END;
				} else { /* we don't check the previous context since the FSM doesn't allow JSON_CONTEXT_KEY */
					state = JSON_COMMA;
				}
			} else {
				if (!token.add(next)) return false;
				state = JSON_KEYWORD;
			}
			break;

		case JSON_COLON:
			if (next == ':') {
				state = JSON_VALUE;
				contexts.last() = JSON_CONTEXT_VALUE;
			} else if (next == ' ' || next == '\t' || next == '\n' || next == '\r') {
				new_line = (next == '\n');
			} else {
				read_error("Expected a colon ':'", current);
				return false;
			}
			break;

		case JSON_COMMA:
			if (next == ',') {
				if (contexts.length == 0) {
					read_error("Unexpected comma ','", current);
					return false;
				}
				if (contexts.last() == JSON_CONTEXT_VALUE) {
					state = JSON_KEY;
					contexts.last() = JSON_CONTEXT_KEY;
				} else { /* we don't check the previous context since the FSM only allows for JSON_CONTEXT_LIST */
					state = JSON_VALUE;
				}
			} else if (next == ' ' || next == '\t' || next == '\n' || next == '\r') {
				new_line = (next == '\n');
			} else if (next == '}') {
				if (contexts.length == 0 || contexts.last() != JSON_CONTEXT_VALUE) {
					read_error("Unexpected closing brace '}'", current);
					return false;
				}
				end_object(current, reader);
				contexts.pop();
				if (contexts.length == 0) {
					state = JSON_END;
				} else { /* we don't check the previous context since the FSM only allows for JSON_CONTEXT_VALUE or JSON_CONTEXT_LIST */
					state = JSON_COMMA;
				}
			} else if (next == ']') {
				if (contexts.length == 0 || contexts.last() != JSON_CONTEXT_LIST) {
					read_error("Unexpected closing brace '}'", current);
					return false;
				}
				end_list(current, reader);
				contexts.pop();
				if (contexts.length == 0) {
					state = JSON_END;
				} else { /* we don't check the previous context since the FSM only allows for JSON_CONTEXT_VALUE or JSON_CONTEXT_LIST */
					state = JSON_COMMA;
				}
			} else {
				read_error("Expected a comma ','", current);
				return false;
			}
			break;

		case JSON_STRING:
			if (next == '"') {
				if (contexts.length == 0) {
					emit_string(token, current, reader);
					state = JSON_END;
				} else if (contexts.last() == JSON_CONTEXT_KEY) {
					emit_key(token, current, reader);
					state = JSON_COLON;
				} else {
					emit_string(token, current, reader);
					state = JSON_COMMA;
				}
				token.clear();
			} else if (next == '\\') {
				next = fgetc(in);
				current.column++;
				emit_escape(in, token, next, current);
			} else {
				if (!token.add(next)) return false;
			}
			break;

		case JSON_KEYWORD:
			if (next == ',') {
				emit_keyword(token, current, reader);
				token.clear();
				if (contexts.length == 0) {
					state = JSON_END;
				} else if (contexts.last() == JSON_CONTEXT_VALUE) {
					state = JSON_KEY;
					contexts.last() = JSON_CONTEXT_KEY;
				} else { /* we don't check the previous context since the FSM only allows for JSON_CONTEXT_LIST */
					state = JSON_VALUE;
				}
			} else if (next == '}') {
				emit_keyword(token, current, reader);
				token.clear();
				if (contexts.length == 0 || contexts.last() != JSON_CONTEXT_VALUE) {
					read_error("Unexpected closing brace '}'", current);
					return false;
				}
				end_object(current, reader);
				contexts.pop();
				if (contexts.length == 0) {
					state = JSON_END;
				} else { /* we don't check the previous context since the FSM only allows for JSON_CONTEXT_VALUE or JSON_CONTEXT_LIST */
					state = JSON_COMMA;
				}
			} else if (next == ']') {
				emit_keyword(token, current, reader);
				token.clear();
				if (contexts.length == 0 || contexts.last() != JSON_CONTEXT_VALUE) {
					read_error("Unexpected comma ','", current);
					return false;
				}
				end_list(current, reader);
				contexts.pop();
				if (contexts.length == 0) {
					state = JSON_END;
				} else { /* we don't check the previous context since the FSM only allows for JSON_CONTEXT_VALUE or JSON_CONTEXT_LIST */
					state = JSON_COMMA;
				}
			} else if (next == ' ' || next == '\t' || next == '\n' || next == '\r') {
				emit_keyword(token, current, reader);
				token.clear();
				if (contexts.length == 0) {
					state = JSON_END;
				} else if (contexts.last() == JSON_CONTEXT_LIST || contexts.last() == JSON_CONTEXT_VALUE) {
					state = JSON_COMMA;
				}
			} else {
				if (!token.add(next)) return false;
			}
			break;

		case JSON_NUMBER:
			if (isdigit(next) || next == '+' || next == '-' || next == '.' || next == 'e' || next == 'E') {
				if (!token.add(next)) return false;
			} else if (next == ',') {
				emit_number(token, current, reader);
				token.clear();
				if (contexts.length == 0) {
					state = JSON_END;
				} else if (contexts.last() == JSON_CONTEXT_VALUE) {
					state = JSON_KEY;
					contexts.last() = JSON_CONTEXT_KEY;
				} else { /* we don't check the previous context since the FSM only allows for JSON_CONTEXT_LIST */
					state = JSON_VALUE;
				}
			} else if (next == '}') {
				emit_number(token, current, reader);
				token.clear();
				if (contexts.length == 0 || contexts.last() != JSON_CONTEXT_VALUE) {
					read_error("Unexpected closing brace '}'", current);
					return false;
				}
				end_object(current, reader);
				contexts.pop();
				if (contexts.length == 0) {
					state = JSON_END;
				} else { /* we don't check the previous context since the FSM only allows for JSON_CONTEXT_VALUE or JSON_CONTEXT_LIST */
					state = JSON_COMMA;
				}
			} else if (next == ']') {
				emit_number(token, current, reader);
				token.clear();
				if (contexts.length == 0 || contexts.last() != JSON_CONTEXT_VALUE) {
					read_error("Unexpected comma ','", current);
					return false;
				}
				end_list(current, reader);
				contexts.pop();
				if (contexts.length == 0) {
					state = JSON_END;
				} else { /* we don't check the previous context since the FSM only allows for JSON_CONTEXT_VALUE or JSON_CONTEXT_LIST */
					state = JSON_COMMA;
				}
			} else if (next == ' ' || next == '\t' || next == '\n' || next == '\r') {
				emit_number(token, current, reader);
				token.clear();
				if (contexts.length == 0) {
					state = JSON_END;
				} else if (contexts.last() == JSON_CONTEXT_LIST || contexts.last() == JSON_CONTEXT_VALUE) {
					state = JSON_COMMA;
				}
			} else {
				read_error("Unexpected symbol inside number", current);
				return false;
			}
			break;

		case JSON_END:
			if (next != ' ' && next != '\t' && next != '\n' && next == '\r') {
				read_error("Unexpected symbol. Expected end of input", current);
				return false;
			}
		}

		if (new_line) {
			current.line++;
			current.column = 1;
			new_line = false;
		} else current.column++;
		next = fgetc(in);
	}

	if (state != JSON_END) {
		read_error("Unexpected end of input", current);
		return false;
	}
	return true;
}

#endif /* JSON_H_ */
