import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'

const certPath = path.resolve(__dirname, 'certs/local-cert.pem')
const keyPath = path.resolve(__dirname, 'certs/local-key.pem')

const hasHttpsCert = existsSync(certPath) && existsSync(keyPath)

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    https: hasHttpsCert
      ? {
          cert: readFileSync(certPath),
          key: readFileSync(keyPath),
        }
      : undefined,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://127.0.0.1:8080',
        ws: true,
      },
    },
  },
})
