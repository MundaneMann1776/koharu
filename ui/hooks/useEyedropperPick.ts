'use client'

import { useEffect, useRef, useCallback } from 'react'
import type React from 'react'
import { convertToImageBitmap } from '@/lib/util'
import { useEditorUiStore } from '@/lib/stores/editorUiStore'
import { usePreferencesStore } from '@/lib/stores/preferencesStore'
import type { PointerToDocumentFn } from '@/hooks/usePointerToDocument'

function rgbaToHex(r: number, g: number, b: number): string {
  return (
    '#' +
    [r, g, b]
      .map((v) => v.toString(16).padStart(2, '0').toUpperCase())
      .join('')
  )
}

type UseEyedropperPickOptions = {
  imageData: Uint8Array | undefined
  documentWidth: number | undefined
  documentHeight: number | undefined
  pointerToDocument: PointerToDocumentFn
  enabled: boolean
}

export function useEyedropperPick({
  imageData,
  documentWidth,
  documentHeight,
  pointerToDocument,
  enabled,
}: UseEyedropperPickOptions) {
  // Cache the ImageBitmap so we don't re-decode on every click.
  // Keyed by the imageData reference — invalidated automatically when imageData changes.
  const bitmapRef = useRef<ImageBitmap | null>(null)
  const lastImageDataRef = useRef<Uint8Array | undefined>(undefined)
  // Guard against concurrent bitmap creation from rapid clicks.
  const decodingRef = useRef(false)

  // Invalidate the cached bitmap when imageData changes.
  useEffect(() => {
    if (lastImageDataRef.current !== imageData) {
      bitmapRef.current?.close()
      bitmapRef.current = null
      decodingRef.current = false
      lastImageDataRef.current = imageData
    }
  }, [imageData])

  // Escape key cancels eyedropper and returns to brush mode.
  useEffect(() => {
    if (!enabled) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        useEditorUiStore.getState().setMode('brush')
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [enabled])

  const handleClick = useCallback(
    async (event: React.MouseEvent<HTMLDivElement>) => {
      if (!enabled) return

      if (!imageData || !documentWidth || !documentHeight) {
        useEditorUiStore.getState().showError('Image not ready — try again in a moment')
        useEditorUiStore.getState().setMode('brush')
        return
      }

      const pos = pointerToDocument(event)
      if (!pos) return

      // Clamp to document bounds.
      const px = Math.floor(Math.max(0, Math.min(pos.x, documentWidth - 1)))
      const py = Math.floor(Math.max(0, Math.min(pos.y, documentHeight - 1)))

      // Get or create the cached ImageBitmap.
      // Guard prevents concurrent decoding from rapid double-clicks.
      if (!bitmapRef.current) {
        if (decodingRef.current) return
        decodingRef.current = true
        try {
          bitmapRef.current = await convertToImageBitmap(imageData)
        } catch (err) {
          decodingRef.current = false
          useEditorUiStore.getState().showError('Failed to decode image for color picking')
          useEditorUiStore.getState().setMode('brush')
          console.error('[eyedropper] bitmap decode failed:', err)
          return
        }
        decodingRef.current = false
      }

      const bitmap = bitmapRef.current

      // Draw the target pixel into a 1×1 OffscreenCanvas using a source crop.
      // This avoids allocating a full-resolution canvas on every click.
      const offscreen = new OffscreenCanvas(1, 1)
      const ctx = offscreen.getContext('2d')
      if (!ctx) {
        useEditorUiStore.getState().showError('Color picking is not supported in this environment')
        useEditorUiStore.getState().setMode('brush')
        return
      }

      ctx.drawImage(bitmap, px, py, 1, 1, 0, 0, 1, 1)
      const pixel = ctx.getImageData(0, 0, 1, 1)
      const [r, g, b] = pixel.data as unknown as [number, number, number]

      const color = rgbaToHex(r, g, b)
      usePreferencesStore.getState().setBrushConfig({ color })
      useEditorUiStore.getState().setMode('brush')
    },
    [enabled, imageData, documentWidth, documentHeight, pointerToDocument],
  )

  return { handleClick }
}
