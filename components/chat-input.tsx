'use client'

import React from 'react'
import { Mic, Send } from 'lucide-react'

interface ChatInputProps {
  input: string
  setInput: (value: string) => void
  onSend: () => void
  isLoading: boolean
  isVoiceActive: boolean
  onVoiceToggle: () => void
}

export default function ChatInput({
  input,
  setInput,
  onSend,
  isLoading,
  isVoiceActive,
  onVoiceToggle,
}: ChatInputProps) {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input && input.trim() && !isLoading) {
      onSend()
      setInput('')
    }
  }

  return (
    <div className="border-t border-accent/20 p-4 backdrop-blur-sm">
      <form onSubmit={handleSubmit} className="flex items-center gap-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about wines, pairings, or recommendations..."
          className="flex-1 bg-input border border-accent/20 rounded-lg px-4 py-3 text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-accent/50 text-sm font-light backdrop-blur-sm"
          disabled={isLoading}
        />

        {/* Voice button with pulsing glow */}
        <button
          type="button"
          onClick={onVoiceToggle}
          className={`relative p-3 rounded-lg transition-all duration-300 ${
            isVoiceActive
              ? 'bg-accent/40 text-foreground border border-accent/60'
              : 'bg-accent/20 text-accent border border-accent/30 hover:bg-accent/30'
          }`}
          title="Voice input"
        >
          {isVoiceActive && (
            <>
              <div className="absolute inset-0 rounded-lg bg-accent/20 animate-pulse" />
              <div className="absolute inset-0 rounded-lg border border-accent/50 animate-pulse" style={{
                animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
              }} />
            </>
          )}
          <Mic size={20} className="relative z-10" />
        </button>

        {/* Send button */}
        <button
          type="submit"
          disabled={isLoading || !input || !input.trim()}
          className="p-3 bg-accent/30 hover:bg-accent/40 disabled:bg-muted/20 disabled:text-muted-foreground text-accent border border-accent/30 rounded-lg transition-all duration-300 disabled:opacity-50"
          title="Send message"
        >
          <Send size={20} />
        </button>
      </form>

      {/* Decorative bottom accent */}
      <div
        className="mt-3 h-px bg-gradient-to-r from-transparent via-accent/20 to-transparent"
      />
    </div>
  )
}
