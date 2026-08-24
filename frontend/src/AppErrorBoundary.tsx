import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { error?: Error };

export class AppErrorBoundary extends Component<Props, State> {
  state: State = {};

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("MomentSeek UI render failed", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <main className="ui-crash-screen" role="alert">
        <section className="ui-crash-card">
          <span className="ui-crash-mark">!</span>
          <div>
            <small>MOMENTSEEK UI RECOVERY</small>
            <h1>页面遇到了一次显示异常</h1>
            <p>服务和检索任务不一定中断。错误已保留在浏览器 Console，可重新加载界面恢复使用。</p>
            <pre>{this.state.error.message || "Unknown render error"}</pre>
            <button type="button" onClick={() => window.location.reload()}>重新加载页面</button>
          </div>
        </section>
      </main>
    );
  }
}
