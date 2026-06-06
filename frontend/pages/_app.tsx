import type { AppProps } from "next/app";

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <div style={{ margin: 0, fontFamily: "system-ui, sans-serif" }}>
      <Component {...pageProps} />
    </div>
  );
}

export default MyApp;
