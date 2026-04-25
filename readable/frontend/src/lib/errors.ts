import axios from "axios";

export const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError<{ detail?: unknown }>(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => {
          if (!item || typeof item !== "object") {
            return "";
          }
          const record = item as { msg?: unknown; loc?: unknown };
          const msg = typeof record.msg === "string" ? record.msg : "";
          if (Array.isArray(record.loc) && record.loc.length > 0) {
            const field = String(record.loc[record.loc.length - 1]);
            return field ? `${field}: ${msg}` : msg;
          }
          return msg;
        })
        .filter(Boolean);
      if (messages.length > 0) {
        return messages.join(" | ");
      }
    }
    if (detail && typeof detail === "object") {
      return "Request failed due to invalid payload.";
    }
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "Something went wrong.";
};
