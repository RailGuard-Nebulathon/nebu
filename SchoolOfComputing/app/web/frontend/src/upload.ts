export const MAX_UPLOAD_BYTES = 30 * 1024 * 1024;
export const MAX_UPLOAD_LABEL = "30 MB";

export function getUploadError(file: Pick<File, "size">): string | null {
  return file.size > MAX_UPLOAD_BYTES
    ? `The file is too large. Choose a file smaller than ${MAX_UPLOAD_LABEL}.`
    : null;
}
