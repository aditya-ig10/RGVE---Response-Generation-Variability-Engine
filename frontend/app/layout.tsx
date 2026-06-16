import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RGVE",
  description: "Response Generation Variation Explorer",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-mono antialiased">{children}</body>
    </html>
  );
}
