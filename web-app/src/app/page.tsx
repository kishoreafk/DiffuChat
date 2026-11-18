'use client'

import { useState } from 'react'
import { LandingView } from '@/components/diffuchat/landing-view'
import { LoginPage } from '@/components/diffuchat/login-view'
import { SignupPage } from '@/components/diffuchat/signup-view'
import { ChatView } from '@/components/diffuchat/chat-view'
import { ReportView } from '@/components/diffuchat/report-view'

type View = 'landing' | 'login' | 'signup' | 'chat' | 'report'

export default function Home() {
  const [view, setView] = useState<View>('landing')
  const [user, setUser] = useState<string | null>(null)

  if (view === 'chat') {
    return <ChatView />
  }
  if (view === 'report') {
    return <ReportView onBack={() => setView('landing')} />
  }
  if (view === 'login') {
    return (
      <LoginPage
        onBack={() => setView('landing')}
        onSignup={() => setView('signup')}
        onSignIn={(email) => {
          setUser(email)
          setView('chat')
        }}
      />
    )
  }
  if (view === 'signup') {
    return (
      <SignupPage
        onBack={() => setView('landing')}
        onSignIn={(email) => {
          setUser(email)
          setView('chat')
        }}
      />
    )
  }
  return (
    <LandingView
      onLogin={() => setView('login')}
      onChat={() => setView(user ? 'chat' : 'login')}
      onReport={() => setView('report')}
    />
  )
}
