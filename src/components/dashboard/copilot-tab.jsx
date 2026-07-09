"use client";

import { useState, useRef, useEffect } from "react";
import { fetchAskAI, fetchDailyReport, fetchUnderperforming } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import ReactMarkdown from "react-markdown";
import { Bot, Send, Loader2, FileText, AlertTriangle, ShoppingCart, Sparkles } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const QUICK_ACTIONS = [
  { label: "Daily Report", icon: FileText, action: "report" },
  { label: "Underperformers", icon: AlertTriangle, action: "underperformers" },
  { label: "Restocking needs", icon: ShoppingCart, action: "restock" },
  { label: "Best products", icon: Sparkles, action: "best" },
];

export function CopilotTab() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const addAssistantMessage = (content, meta) => {
    setMessages((prev) => [...prev, { role: "assistant", content, ...meta }]);
  };

  const handleSend = async (question) => {
    const q = question || input.trim();
    if (!q || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetchAskAI({ question: q });
      addAssistantMessage(res.answer, {
        sources: res.sources,
        latency_ms: res.latency_ms,
        mode: res.mode,
      });
    } catch (err) {
      addAssistantMessage("Error: " + err.message, {});
    } finally {
      setLoading(false);
    }
  };

  const handleQuickAction = async (action) => {
    if (loading) return;

    if (action === "report") {
      setMessages((prev) => [...prev, { role: "user", content: "Generate daily intelligence report" }]);
      setLoading(true);
      try {
        const res = await fetchDailyReport();
        const report = typeof res.report === "string" ? res.report : JSON.stringify(res.report, null, 2);
        addAssistantMessage(report, { latency_ms: res.latency_ms, mode: "report" });
      } catch (err) {
        addAssistantMessage("Error: " + err.message, {});
      } finally {
        setLoading(false);
      }
    } else if (action === "underperformers") {
      setMessages((prev) => [...prev, { role: "user", content: "Which products are underperforming?" }]);
      setLoading(true);
      try {
        const res = await fetchUnderperforming(null, 10);
        let text = "## Underperforming Products\n\n";
        text += "Found **" + res.total_underperforming + "** underperforming products:\n\n";
        res.products.forEach((p, i) => {
          const actionIcon = p.action === "delist" ? "❌" : p.action === "promote" ? "📢" : "👁️";
          text += i + 1 + ". **" + p.item_id + "** (" + p.store_id + ") — " + p.action.toUpperCase() + "\n";
          text += "   Avg sales: " + p.avg_daily_sales + "/day. " + p.reasoning + "\n\n";
        });
        if (res.ai_summary) text += "\n---\n" + res.ai_summary;
        addAssistantMessage(text, { latency_ms: res.latency_ms, mode: "underperformer" });
      } catch (err) {
        addAssistantMessage("Error: " + err.message, {});
      } finally {
        setLoading(false);
      }
    } else if (action === "restock") {
      setInput("Which products need restocking across all stores?");
    } else if (action === "best") {
      setInput("What are the top 5 best selling products?");
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-220px)] min-h-[400px]">
      {/* Chat Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto pr-4">
        <div className="space-y-4 pb-4">
          {messages.length === 0 && (
            <div className="text-center py-12 text-muted-foreground">
              <Bot className="h-12 w-12 mx-auto mb-3 opacity-40" />
              <p className="font-medium">Retail AI Copilot</p>
              <p className="text-sm mt-1">Ask anything about your retail data</p>
            </div>
          )}

          <AnimatePresence>
            {messages.map((msg, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={msg.role === "user" ? "flex justify-end" : ""}
              >
                {msg.role === "user" ? (
                  <div className="max-w-[70%] bg-primary text-primary-foreground rounded-2xl rounded-br-md px-4 py-2.5">
                    <p className="text-sm">{msg.content}</p>
                  </div>
                ) : (
                  <div className="max-w-[85%] space-y-2">
                    <div className="bg-muted rounded-2xl rounded-bl-md px-4 py-3">
                      <div className="prose prose-sm dark:prose-invert max-w-none [&_p]:mb-2 [&_li]:mb-1">
                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground pl-2">
                      {msg.mode && <Badge variant="outline" className="text-[10px] h-5">{msg.mode}</Badge>}
                      {msg.latency_ms && <span>{msg.latency_ms}ms</span>}
                      {msg.sources?.length > 0 && <span>{msg.sources.length} sources</span>}
                    </div>
                  </div>
                )}
              </motion.div>
            ))}
          </AnimatePresence>

          {loading && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex justify-start">
              <div className="bg-muted rounded-2xl rounded-bl-md px-4 py-3">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" /> Thinking...
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>

      <Separator className="my-2" />

      {/* Quick Actions */}
      <div className="flex flex-wrap gap-2 mb-3">
        {QUICK_ACTIONS.map(({ label, icon: I, action }) => (
          <Button key={action} variant="outline" size="sm" onClick={() => handleQuickAction(action)} disabled={loading}>
            <I className="h-3.5 w-3.5 mr-1" />{label}
          </Button>
        ))}
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about your retail data..."
          className="min-h-[44px] max-h-[120px] resize-none"
          rows={1}
          disabled={loading}
        />
        <Button onClick={() => handleSend()} disabled={!input.trim() || loading} size="icon" className="shrink-0 h-[44px] w-[44px]">
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}