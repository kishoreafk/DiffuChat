import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as SonnerToaster } from "@/components/ui/sonner";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DiffuChat — chat with a model that writes in any order",
  description:
    "DiffuChat playground: a masked-diffusion language model trained from scratch on Kaggle, denoising answers in parallel and letting you pin-edit any token. Includes the architecture-matched AR baseline report.",
  keywords: ["DiffuChat", "diffusion LM", "masked diffusion", "DenoiseLM", "Kaggle", "MDLM"],
  icons: {
    icon: "https://z-cdn.chatglm.cn/z-ai/static/logo.svg",
  },
  openGraph: {
    title: "DiffuChat",
    description: "Watch answers denoise — then edit any token.",
    siteName: "DiffuChat",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
        <SonnerToaster position="bottom-right" richColors />
      </body>
    </html>
  );
}
