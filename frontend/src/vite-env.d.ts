/// <reference types="vite/client" />

// Declare CSS module types for side-effect imports
declare module '*.css' {
  const content: Record<string, string>
  export default content
}
