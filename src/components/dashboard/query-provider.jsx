"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function QueryProvider({ children }) {
  const [client] = useState(() => new QueryClient({
    defaultOptions: {
      queries: { staleTime: 60000, retry: 1, refetchOnWindowFocus: false },
    },
  }));

  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}