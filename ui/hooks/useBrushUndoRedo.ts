'use client'

import { useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useBrushHistoryStore } from '@/lib/stores/brushHistoryStore'
import { useEditorUiStore } from '@/lib/stores/editorUiStore'
import {
  getGetDocumentQueryKey,
  getListDocumentsQueryKey,
} from '@/lib/api/documents/documents'
import { useTextBlocks } from '@/hooks/useTextBlocks'
import { fetchApi } from '@/lib/api/fetch'

type BrushUndoRedoOptions = {
  enabled: boolean
}

export function useBrushUndoRedo({ enabled }: BrushUndoRedoOptions) {
  const queryClient = useQueryClient()
  const { document: currentDoc } = useTextBlocks()
  const currentDocumentId = useEditorUiStore((s) => s.currentDocumentId)
  const mode = useEditorUiStore((s) => s.mode)
  const isProcessingRef = useRef(false)

  const canUndo = useBrushHistoryStore((s) =>
    currentDocumentId ? s.canUndo(currentDocumentId) : false,
  )
  const canRedo = useBrushHistoryStore((s) =>
    currentDocumentId ? s.canRedo(currentDocumentId) : false,
  )

  const handleUndo = async () => {
    if (!currentDocumentId || !currentDoc || isProcessingRef.current) return
    if (!canUndo) return

    isProcessingRef.current = true
    try {
      const historyEntry = useBrushHistoryStore
        .getState()
        .undo(currentDocumentId)
      if (!historyEntry) return

      // Call replace_brush_layer API with full PNG
      // Convert Uint8Array to proper ArrayBuffer for fetch body
      const buffer = historyEntry.brushLayerPng.buffer.slice(
        historyEntry.brushLayerPng.byteOffset,
        historyEntry.brushLayerPng.byteOffset +
          historyEntry.brushLayerPng.byteLength,
      ) as ArrayBuffer
      await fetchApi<void>(
        `/api/v1/documents/${currentDocumentId}/brush-layer-replace`,
        {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/octet-stream',
          },
          body: buffer,
        },
      )

      // Invalidate cache to refresh UI
      await queryClient.invalidateQueries({
        queryKey: getGetDocumentQueryKey(currentDocumentId),
      })
      await queryClient.invalidateQueries({
        queryKey: getListDocumentsQueryKey(),
      })
    } catch (error) {
      console.error('Undo failed:', error)
      useEditorUiStore.getState().showError('Failed to undo brush operation')
    } finally {
      isProcessingRef.current = false
    }
  }

  const handleRedo = async () => {
    if (!currentDocumentId || !currentDoc || isProcessingRef.current) return
    if (!canRedo) return

    isProcessingRef.current = true
    try {
      const historyEntry = useBrushHistoryStore
        .getState()
        .redo(currentDocumentId)
      if (!historyEntry) return

      // Call replace_brush_layer API with full PNG
      // Convert Uint8Array to proper ArrayBuffer for fetch body
      const buffer = historyEntry.brushLayerPng.buffer.slice(
        historyEntry.brushLayerPng.byteOffset,
        historyEntry.brushLayerPng.byteOffset +
          historyEntry.brushLayerPng.byteLength,
      ) as ArrayBuffer
      await fetchApi<void>(
        `/api/v1/documents/${currentDocumentId}/brush-layer-replace`,
        {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/octet-stream',
          },
          body: buffer,
        },
      )

      // Invalidate cache to refresh UI
      await queryClient.invalidateQueries({
        queryKey: getGetDocumentQueryKey(currentDocumentId),
      })
      await queryClient.invalidateQueries({
        queryKey: getListDocumentsQueryKey(),
      })
    } catch (error) {
      console.error('Redo failed:', error)
      useEditorUiStore.getState().showError('Failed to redo brush operation')
    } finally {
      isProcessingRef.current = false
    }
  }

  useEffect(() => {
    if (!enabled) return

    const handler = (e: KeyboardEvent) => {
      // Only handle in brush/eraser mode
      if (mode !== 'brush' && mode !== 'eraser') return

      // Check for Cmd (macOS) or Ctrl (Windows/Linux)
      const isMac =
        navigator.platform.includes('Mac') ||
        navigator.userAgent.includes('Mac')
      const modifier = isMac ? e.metaKey : e.ctrlKey

      if (!modifier || e.key !== 'z') return

      // Prevent if user is typing in an input/textarea
      const target = e.target as HTMLElement
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return
      }

      if (e.shiftKey) {
        // Cmd+Shift+Z = Redo
        e.preventDefault()
        void handleRedo()
      } else {
        // Cmd+Z = Undo
        e.preventDefault()
        void handleUndo()
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [enabled, mode, currentDocumentId, canUndo, canRedo])

  return {
    canUndo,
    canRedo,
    undo: handleUndo,
    redo: handleRedo,
  }
}
