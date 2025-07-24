import { useToast } from "@/hooks/use-toast"
import {
  Toast,
  ToastClose,
  ToastDescription,
  ToastProvider,
  ToastTitle,
  ToastViewport,
} from "@/components/ui/toast"

export function Toaster() {
  const { toasts } = useToast()

  return (
    <ToastProvider>
      {toasts.map(function ({ id, title, description, action, ...props }) {
        // Split description by \n for multi-line support
        const descLines = typeof description === 'string' ? description.split('\n') : [description];
        return (
          <Toast key={id} {...props}>
            <div className="grid gap-1">
              {title && (
                <ToastTitle className="text-2xl font-extrabold">{title}</ToastTitle>
              )}
              {descLines.map((line, idx) => (
                <ToastDescription key={idx} className="text-lg font-bold leading-snug">{line}</ToastDescription>
              ))}
            </div>
            {action}
            <ToastClose />
          </Toast>
        )
      })}
      <ToastViewport />
    </ToastProvider>
  )
}
