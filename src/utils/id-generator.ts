import { v4 as uuidv4 } from 'uuid';

/**
 * Generate a unique ID for follicles
 */
export function generateId(): string {
  return uuidv4();
}
