import React from 'react'

interface ChatContainerProps {
  children: React.ReactNode
}

export default function ChatContainer({ children }: ChatContainerProps) {
  return (
    <div className="w-full max-w-2xl">
      {/* Glassmorphism container */}
      <div
        className="backdrop-blur-md rounded-2xl border border-accent/20 shadow-2xl overflow-hidden flex flex-col h-[600px] bg-card"
        style={{
          background: 'rgba(20, 10, 30, 0.25)',
          backdropFilter: 'blur(10px)',
          boxShadow: '0 8px 32px 0 rgba(196, 164, 92, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.05)',
        }}
      >
        {children}
      </div>

      {/* Decorative glow effect */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(circle at 50% 50%, rgba(196, 164, 92, 0.05) 0%, transparent 70%)',
          filter: 'blur(40px)',
          zIndex: -1,
        }}
      />
    </div>
  )
}
