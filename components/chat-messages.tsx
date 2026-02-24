import React, { useEffect, useRef } from 'react'

interface ChatMessagesProps {
  messages: ChatMessage[]
  isLoading: boolean
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
}

export default function ChatMessages({ messages, isLoading }: ChatMessagesProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
        <div className="text-5xl mb-4 font-serif text-accent">🍷</div>
        <h1 className="text-3xl font-serif text-foreground mb-2">Wine Sommelier</h1>
        <p className="text-muted-foreground font-light">
          Welcome to your personal AI wine expert. Ask anything about wines, pairings, or recommendations.
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-accent/30 scrollbar-track-transparent">
      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-xs lg:max-w-md px-4 py-3 rounded-lg backdrop-blur-sm ${
              message.role === 'user'
                ? 'bg-accent/20 text-foreground border border-accent/30 rounded-br-none'
                : 'bg-primary/20 text-foreground border border-primary/30 rounded-bl-none'
            }`}
          >
            <p className="text-sm leading-relaxed font-light">{message.text}</p>
          </div>
        </div>
      ))}

      {isLoading && messages[messages.length - 1]?.text === '' && (
        <div className="flex justify-start">
          <div className="bg-primary/20 border border-primary/30 rounded-lg rounded-bl-none px-4 py-3 backdrop-blur-sm">
            <div className="flex space-x-2">
              <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0s' }} />
              <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
              <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  )
}
