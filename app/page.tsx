'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import ChatContainer from '@/components/chat-container'
import ChatMessages from '@/components/chat-messages'
import type { ChatMessage } from '@/components/chat-messages'
import ChatInput from '@/components/chat-input'

const RENDER_HOST = process.env.NEXT_PUBLIC_RENDER_BACKEND_HOST
const BACKEND_WS_URL = process.env.NEXT_PUBLIC_BACKEND_WS_URL
  ?? (RENDER_HOST ? `wss://${RENDER_HOST}/ws/chat` : 'ws://localhost:8001/ws/chat')

function toFloat32FromPcm16(base64Pcm: string): Float32Array {
  const binary = atob(base64Pcm)
  const length = binary.length / 2
  const buffer = new Float32Array(length)
  for (let i = 0; i < length; i += 1) {
    const lo = binary.charCodeAt(i * 2)
    const hi = binary.charCodeAt(i * 2 + 1)
    let sample = (hi << 8) | lo
    if (sample >= 0x8000) sample -= 0x10000
    buffer[i] = sample / 0x8000
  }
  return buffer
}

function resampleLinear(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return input
  const ratio = toRate / fromRate
  const outputLength = Math.max(1, Math.round(input.length * ratio))
  const output = new Float32Array(outputLength)
  for (let i = 0; i < outputLength; i += 1) {
    const sourcePos = i / ratio
    const left = Math.floor(sourcePos)
    const right = Math.min(left + 1, input.length - 1)
    const mix = sourcePos - left
    output[i] = input[left] * (1 - mix) + input[right] * mix
  }
  return output
}

function float32ToPcm16Base64(input: Float32Array): string {
  const pcm = new Int16Array(input.length)
  for (let i = 0; i < input.length; i += 1) {
    const value = Math.max(-1, Math.min(1, input[i]))
    pcm[i] = value < 0 ? value * 0x8000 : value * 0x7fff
  }

  const bytes = new Uint8Array(pcm.buffer)
  let binary = ''
  const step = 0x8000
  for (let i = 0; i < bytes.length; i += step) {
    const chunk = bytes.subarray(i, i + step)
    binary += String.fromCharCode(...chunk)
  }
  return btoa(binary)
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isVoiceActive, setIsVoiceActive] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const wsReadyRef = useRef(false)
  const sessionIdRef = useRef('')
  const currentAssistantIdRef = useRef<string | null>(null)
  const currentUserIdRef = useRef<string | null>(null)

  const audioContextRef = useRef<AudioContext | null>(null)
  const audioCursorRef = useRef(0)

  const micStreamRef = useRef<MediaStream | null>(null)
  const micContextRef = useRef<AudioContext | null>(null)
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const micProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const micMuteGainRef = useRef<GainNode | null>(null)
  const micLastSpeechAtRef = useRef(0)
  const micStartedAtRef = useRef(0)
  const micHeardSpeechRef = useRef(false)
  const micAutoStoppingRef = useRef(false)
  const micStoppingRef = useRef(false)

  const ensureSessionId = useCallback(() => {
    if (sessionIdRef.current) return sessionIdRef.current
    const storageKey = 'wine-bot-session-id'
    const existing = window.localStorage.getItem(storageKey)
    if (existing) {
      sessionIdRef.current = existing
      return existing
    }
    const created = crypto.randomUUID()
    window.localStorage.setItem(storageKey, created)
    sessionIdRef.current = created
    return created
  }, [])

  const playAudioChunk = useCallback(async (base64Pcm: string, sampleRate: number) => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext()
      audioCursorRef.current = audioContextRef.current.currentTime
    }

    const ctx = audioContextRef.current
    if (ctx.state === 'suspended') {
      await ctx.resume()
    }

    const decoded = toFloat32FromPcm16(base64Pcm)
    const rendered = resampleLinear(decoded, sampleRate, ctx.sampleRate)
    const audioBuffer = ctx.createBuffer(1, rendered.length, ctx.sampleRate)
    audioBuffer.copyToChannel(new Float32Array(rendered), 0)

    const source = ctx.createBufferSource()
    source.buffer = audioBuffer
    source.connect(ctx.destination)

    const startAt = Math.max(audioCursorRef.current, ctx.currentTime + 0.05)
    source.start(startAt)
    audioCursorRef.current = startAt + audioBuffer.duration
  }, [])

  const handleSocketMessage = useCallback(
    async (event: MessageEvent<string>) => {
      const payload = JSON.parse(event.data)
      const type = payload.type as string

      if (type === 'session_ready' || type === 'pong' || type === 'audio_started' || type === 'session_reset') {
        return
      }

      if (type === 'no_speech_detected') {
        const userId = currentUserIdRef.current
        const assistantId = currentAssistantIdRef.current
        setMessages((prev) =>
          prev.filter(
            (msg) =>
              !(
                ((userId && msg.id === userId) || (assistantId && msg.id === assistantId)) &&
                msg.text.trim().length === 0
              )
          )
        )
        setIsLoading(false)
        currentAssistantIdRef.current = null
        currentUserIdRef.current = null
        return
      }

      if (type === 'user_transcript_delta') {
        const delta = String(payload.text ?? '')
        let userId = currentUserIdRef.current
        let assistantId = currentAssistantIdRef.current

        if (!userId || !assistantId) {
          const newUserId = crypto.randomUUID()
          const newAssistantId = crypto.randomUUID()
          currentUserIdRef.current = newUserId
          currentAssistantIdRef.current = newAssistantId
          userId = newUserId
          assistantId = newAssistantId
          setMessages((prev) => [
            ...prev,
            { id: newUserId, role: 'user', text: '' },
            { id: newAssistantId, role: 'assistant', text: '' },
          ])
        }

        setMessages((prev) =>
          prev.map((msg) => (msg.id === userId ? { ...msg, text: msg.text + delta } : msg))
        )
        return
      }

      if (type === 'assistant_text_delta') {
        const delta = String(payload.text ?? '')
        let userId = currentUserIdRef.current
        let assistantId = currentAssistantIdRef.current

        if (!userId || !assistantId) {
          const newUserId = crypto.randomUUID()
          const newAssistantId = crypto.randomUUID()
          currentUserIdRef.current = newUserId
          currentAssistantIdRef.current = newAssistantId
          userId = newUserId
          assistantId = newAssistantId
          setMessages((prev) => [
            ...prev,
            { id: newUserId, role: 'user', text: '' },
            { id: newAssistantId, role: 'assistant', text: '' },
          ])
        }

        setMessages((prev) =>
          prev.map((msg) => (msg.id === assistantId ? { ...msg, text: msg.text + delta } : msg))
        )
        return
      }

      if (type === 'assistant_audio_chunk') {
        const audioBase64 = String(payload.audio_base64 ?? '')
        const sampleRate = Number(payload.sample_rate ?? 24000)
        if (audioBase64) {
          await playAudioChunk(audioBase64, sampleRate)
        }
        return
      }

      if (type === 'assistant_turn_end') {
        setIsLoading(false)
        currentAssistantIdRef.current = null
        currentUserIdRef.current = null
      }
    },
    [playAudioChunk]
  )

  const ensureSocket = useCallback(async () => {
    if (wsRef.current && wsReadyRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return wsRef.current
    }

    const sessionId = ensureSessionId()
    const socket = new WebSocket(`${BACKEND_WS_URL}?session_id=${encodeURIComponent(sessionId)}`)
    wsRef.current = socket
    wsReadyRef.current = false

    await new Promise<void>((resolve, reject) => {
      socket.onopen = () => {
        wsReadyRef.current = true
        resolve()
      }
      socket.onerror = () => reject(new Error('Socket failed to connect'))
    })

    socket.onmessage = (evt) => {
      void handleSocketMessage(evt as MessageEvent<string>)
    }

    socket.onclose = () => {
      wsReadyRef.current = false
    }

    return socket
  }, [ensureSessionId, handleSocketMessage])

  const sendUserText = useCallback(
    async (text: string) => {
      const clean = text.trim()
      if (!clean) return

      const socket = await ensureSocket()
      if (isLoading) {
        socket.send(JSON.stringify({ type: 'reset' }))
      }
      const userId = crypto.randomUUID()
      const assistantId = crypto.randomUUID()
      currentUserIdRef.current = userId
      currentAssistantIdRef.current = assistantId

      setMessages((prev) => [
        ...prev,
        { id: userId, role: 'user', text: clean },
        { id: assistantId, role: 'assistant', text: '' },
      ])
      setIsLoading(true)
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext()
      }
      if (audioContextRef.current.state === 'suspended') {
        void audioContextRef.current.resume()
      }
      audioCursorRef.current = audioContextRef.current.currentTime
      socket.send(JSON.stringify({ type: 'user_text', text: clean }))
    },
    [ensureSocket, isLoading]
  )

  const stopVoiceCapture = useCallback(() => {
    if (micStoppingRef.current) return
    micStoppingRef.current = true

    micProcessorRef.current?.disconnect()
    micSourceRef.current?.disconnect()
    micMuteGainRef.current?.disconnect()

    micProcessorRef.current = null
    micSourceRef.current = null
    micMuteGainRef.current = null

    if (micContextRef.current && micContextRef.current.state !== 'closed') {
      void micContextRef.current.close()
    }
    micContextRef.current = null

    micStreamRef.current?.getTracks().forEach((track) => track.stop())
    micStreamRef.current = null

    const socket = wsRef.current
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'user_audio_end' }))
    }

    setIsVoiceActive(false)
    micAutoStoppingRef.current = false
    micStoppingRef.current = false
  }, [])

  const startVoiceCapture = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof AudioContext === 'undefined') {
      alert('Live microphone streaming is not supported in this browser.')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      micStreamRef.current = stream

      const micContext = new AudioContext({ latencyHint: 'interactive' })
      micContextRef.current = micContext
      if (micContext.state === 'suspended') {
        await micContext.resume()
      }

      const socket = await ensureSocket()
      if (isLoading) {
        socket.send(JSON.stringify({ type: 'reset' }))
      }
      const userId = crypto.randomUUID()
      const assistantId = crypto.randomUUID()
      currentUserIdRef.current = userId
      currentAssistantIdRef.current = assistantId

      setMessages((prev) => [
        ...prev,
        { id: userId, role: 'user', text: '' },
        { id: assistantId, role: 'assistant', text: '' },
      ])

      setIsLoading(true)
      setIsVoiceActive(true)
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext()
      }
      if (audioContextRef.current.state === 'suspended') {
        void audioContextRef.current.resume()
      }
      audioCursorRef.current = audioContextRef.current.currentTime
      micLastSpeechAtRef.current = Date.now()
      micStartedAtRef.current = Date.now()
      micHeardSpeechRef.current = false
      micAutoStoppingRef.current = false
      micStoppingRef.current = false
      socket.send(JSON.stringify({ type: 'user_audio_start' }))

      const source = micContext.createMediaStreamSource(stream)
      const processor = micContext.createScriptProcessor(1024, 1, 1)
      const muteGain = micContext.createGain()
      muteGain.gain.value = 0

      micSourceRef.current = source
      micProcessorRef.current = processor
      micMuteGainRef.current = muteGain

      processor.onaudioprocess = (evt: AudioProcessingEvent) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
        const pcmChunk = evt.inputBuffer.getChannelData(0)

        let sumSquares = 0
        for (let i = 0; i < pcmChunk.length; i += 1) {
          const v = pcmChunk[i]
          sumSquares += v * v
        }
        const rms = Math.sqrt(sumSquares / Math.max(pcmChunk.length, 1))
        const now = Date.now()
        if (rms > 0.012) {
          micLastSpeechAtRef.current = now
          micHeardSpeechRef.current = true
        }

        const audioBase64 = float32ToPcm16Base64(pcmChunk)
        wsRef.current.send(
          JSON.stringify({
            type: 'user_audio_chunk',
            audio_base64: audioBase64,
            sample_rate: micContext.sampleRate,
          })
        )

      }

      source.connect(processor)
      processor.connect(muteGain)
      muteGain.connect(micContext.destination)
    } catch (error) {
      stopVoiceCapture()
      setIsLoading(false)
      currentAssistantIdRef.current = null
      currentUserIdRef.current = null
      console.error(error)
    }
  }, [ensureSocket, isLoading, stopVoiceCapture])

  const handleVoiceToggle = useCallback(async () => {
    if (isVoiceActive) {
      stopVoiceCapture()
      return
    }
    await startVoiceCapture()
  }, [isVoiceActive, startVoiceCapture, stopVoiceCapture])

  const startNewSession = useCallback(async () => {
    stopVoiceCapture()
    const socket = await ensureSocket()
    socket.send(JSON.stringify({ type: 'reset' }))

    setMessages([])
    setInputValue('')
    setIsLoading(false)
    setIsVoiceActive(false)
    currentAssistantIdRef.current = null
    currentUserIdRef.current = null
  }, [ensureSocket, stopVoiceCapture])

  useEffect(() => {
    void ensureSocket()
    return () => {
      stopVoiceCapture()
      wsRef.current?.close()
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        void audioContextRef.current.close()
      }
    }
  }, [ensureSocket, stopVoiceCapture])

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div
        className="absolute inset-0 bg-cover bg-center brightness-50"
        style={{
          backgroundImage: 'url(/wine-cellar-bg.jpg)',
          filter: 'blur(3px)',
        }}
      />

      <div className="absolute inset-0 bg-black/40" />

      <div className="relative z-10 flex items-center justify-center min-h-screen p-4">
        <ChatContainer>
          <div className="flex justify-end px-4 py-3 border-b border-accent/20">
            <button
              type="button"
              onClick={() => {
                void startNewSession()
              }}
              className="px-3 py-1.5 text-xs rounded-md border border-accent/40 text-accent bg-accent/10 hover:bg-accent/20 transition-colors"
            >
              New Session
            </button>
          </div>
          <ChatMessages messages={messages} isLoading={isLoading} />
          <ChatInput
            input={inputValue}
            setInput={setInputValue}
            onSend={() => {
              void sendUserText(inputValue)
              setInputValue('')
            }}
            isLoading={isLoading}
            isVoiceActive={isVoiceActive}
            onVoiceToggle={() => {
              void handleVoiceToggle()
            }}
          />
        </ChatContainer>
      </div>
    </div>
  )
}
