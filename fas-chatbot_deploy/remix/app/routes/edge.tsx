import type { MetaFunction } from "@vercel/remix";

export const config = { runtime: "edge" };

export const meta: MetaFunction = () => [
  { title: "Chatbot Remix@Edge App" }, // Update the title
  { name: "description", content: "Chat with the Remix Chatbot at the Edge!" }, // Update the description
];

export default function Edge() {
  return (
    <div style={{ fontFamily: "system-ui, sans-serif", lineHeight: "1.4" }}>
      <h1>Chat with the Remix Chatbot at the Edge</h1> {/* Update the header text */}
    </div>
  );
}
