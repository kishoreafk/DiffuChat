'use client'

import { useState } from 'react'
import { Eye, EyeOff, ArrowLeft, Waves } from 'lucide-react'

export function LoginPage({
  onBack,
  onSignup,
  onSignIn,
}: {
  onBack: () => void
  onSignup: () => void
  onSignIn: (email: string) => void
}) {
  const [showPassword, setShowPassword] = useState(false)
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    rememberMe: false,
  })

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSignIn(formData.email)
  }

  return (
    <div className="h-screen w-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800 flex">
      {/* Left Panel - Image Section */}
      <div className="flex-1 relative overflow-hidden">
        {/* Back Button */}
        <div className="absolute top-6 left-6 z-10">
          <button
            onClick={onBack}
            aria-label="Go back"
            className="w-10 h-10 bg-black/20 backdrop-blur-sm rounded-full flex items-center justify-center hover:bg-black/30 transition-all"
          >
            <ArrowLeft className="w-5 h-5 text-white" />
          </button>
        </div>

        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800">
          {/* DiffuChat denoising visual (brand adaptation of the split-panel image) */}
          <div className="absolute inset-0 flex flex-col items-center justify-center px-10">
            <div className="w-full max-w-md space-y-3">
              {[
                { label: 'you :', text: 'how do diffusion models write?', dim: false },
                { label: 'ai :', text: 'they start from noise ▓▓▓▓▓▓▓▓▓▓▓ and denoise', dim: true },
              ].map((row, i) => (
                <div
                  key={i}
                  className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm p-4"
                >
                  <p className="text-xs font-medium text-blue-300 mb-1">{row.label}</p>
                  <p className="text-sm text-slate-200 leading-relaxed">
                    {i === 1 ? (
                      <>
                        they start from noise{' '}
                        <span className="inline-flex gap-0.5 align-middle">
                          {'▓▓▓▓▓▓▓▓'.split('').map((c, j) => (
                            <span
                              key={j}
                              className="inline-block w-2.5 h-4 rounded-[2px] bg-blue-400/40 animate-pulse"
                              style={{ animationDelay: `${j * 0.12}s` }}
                            />
                          ))}
                        </span>{' '}
                        and denoise
                      </>
                    ) : (
                      row.text
                    )}
                  </p>
                </div>
              ))}
              <div className="rounded-2xl border border-blue-400/20 bg-blue-500/10 backdrop-blur-sm p-4">
                <p className="text-xs font-medium text-blue-300 mb-1 flex items-center gap-1.5">
                  <Waves className="w-3.5 h-3.5" /> parallel denoising
                </p>
                <p className="text-sm text-slate-200 leading-relaxed">
                  Answers materialize <span className="text-white font-medium">in any order</span> —
                  not left to right. Edit any token; the rest re-flows around it.
                </p>
              </div>
            </div>
            <p className="mt-8 text-xs text-slate-400 tracking-wide uppercase">
              DiffuChat · masked-diffusion language model
            </p>
          </div>
        </div>
      </div>

      {/* Right Panel - Form Section */}
      <div className="flex-1 flex items-center justify-center bg-white">
        <div className="w-full max-w-md p-8">
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Welcome Back</h1>
            <p className="text-gray-600">
              Don&apos;t have an account?{' '}
              <button
                onClick={onSignup}
                className="text-blue-600 hover:text-blue-700 font-medium"
              >
                Sign up
              </button>
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Email */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                placeholder="Email Address"
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none"
                required
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="password"
                  value={formData.password}
                  onChange={handleInputChange}
                  placeholder="Password"
                  className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 hover:bg-gray-100 rounded-full"
                >
                  {showPassword ? (
                    <EyeOff className="w-5 h-5 text-gray-500" />
                  ) : (
                    <Eye className="w-5 h-5 text-gray-500" />
                  )}
                </button>
              </div>
            </div>

            {/* Remember Me + Forgot Password */}
            <div className="flex items-center justify-between">
              <label className="flex items-center space-x-2 text-sm text-gray-600">
                <input
                  type="checkbox"
                  name="rememberMe"
                  checked={formData.rememberMe}
                  onChange={handleInputChange}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded"
                />
                <span>Remember me</span>
              </label>
              <button
                type="button"
                className="text-sm text-blue-600 hover:text-blue-700 font-medium"
              >
                Forgot password?
              </button>
            </div>

            {/* Submit */}
            <button
              type="submit"
              className="w-full bg-black text-white py-3 px-4 rounded-xl font-medium hover:bg-gray-800 transition-colors"
            >
              Sign In
            </button>

            {/* Divider */}
            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-300"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-2 bg-white text-gray-500">or</span>
              </div>
            </div>

            {/* Social Buttons */}
            <div className="grid grid-cols-2 gap-4">
              <button
                type="button"
                onClick={() => onSignIn('demo@google.oauth')}
                className="flex items-center justify-center px-4 py-3 border border-gray-300 rounded-xl hover:bg-gray-50"
              >
                {/* Google SVG */}
                <svg className="w-5 h-5 mr-2" viewBox="0 0 24 24">
                  <path
                    fill="#4285F4"
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  />
                  <path
                    fill="#34A853"
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  />
                  <path
                    fill="#FBBC05"
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  />
                  <path
                    fill="#EA4335"
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  />
                </svg>
                <span className="text-sm font-medium text-gray-700">Google</span>
              </button>

              <button
                type="button"
                onClick={() => onSignIn('demo@github.oauth')}
                className="flex items-center justify-center px-4 py-3 border border-gray-300 rounded-xl hover:bg-gray-50"
              >
                {/* GitHub SVG */}
                <svg className="w-5 h-5 mr-2" fill="#24292f" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                </svg>
                <span className="text-sm font-medium text-gray-700">GitHub</span>
              </button>
            </div>

            <p className="text-xs text-center text-gray-400">
              Demo build — sign-in opens the DiffuChat playground without credential checks.
            </p>
          </form>
        </div>
      </div>
    </div>
  )
}
