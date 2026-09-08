import { Component, type ErrorInfo, type ReactNode } from "react"

/**
 * Menangkap error render agar aplikasi TIDAK menampilkan layar putih kosong.
 * Jika terjadi error, pesan teknis ditampilkan (juga muncul di Console browser).
 * Sengaja memakai inline style agar tetap tampil meski CSS/Tailwind gagal dimuat.
 */
interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Tampil di Console (F12) untuk memudahkan debugging.
    console.error("[UI ErrorBoundary]", error, info)
  }

  render() {
    const { error } = this.state
    if (error) {
      return (
        <div
          style={{
            maxWidth: 680,
            margin: "40px auto",
            padding: 24,
            fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif",
            lineHeight: 1.5,
          }}
        >
          <h1 style={{ fontSize: 20, fontWeight: 700, marginBottom: 8 }}>
            Terjadi kesalahan saat menampilkan aplikasi
          </h1>
          <p style={{ color: "#666", marginBottom: 12 }}>
            Halaman gagal dirender. Detail teknis di bawah ini (juga muncul di
            Console browser — tekan F12 → tab Console).
          </p>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              background: "#f6f6f7",
              color: "#b00020",
              padding: 12,
              borderRadius: 8,
              fontSize: 12,
              border: "1px solid #eee",
            }}
          >
            {String(error.stack || error.message || error)}
          </pre>
          <p style={{ color: "#666", marginTop: 12, fontSize: 13 }}>
            Jika ini muncul setelah pembaruan file, jalankan{" "}
            <code>npm install</code> lalu <code>npm run dev</code> ulang, dan
            muat ulang halaman (Ctrl/Cmd+Shift+R).
          </p>
        </div>
      )
    }
    return this.props.children
  }
}
