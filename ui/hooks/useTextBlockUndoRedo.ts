'use client'

import { useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { putTextBlocks } from '@/lib/api/text-blocks/text-blocks'
import {
  getGetDocumentQueryKey,
  getListDocumentsQueryKey,
} from '@/lib/api/documents/documents'
import { useEditorUiStore } from '@/lib/stores/editorUiStore'
import { useTextBlockHistoryStore } from '@/lib/stores/textBlockHistoryStore'
import { useCurrentDocument } from '@/hooks/useTextBlocks'

type Options = {
  enabled: boolean
}

export function useTextBlockUndoRedo({ enabled }: Options) {
  const queryClient = useQueryClient()
  const currentDocument = useCurrentDocument()
  const currentDocumentId = useEditorUiStore((s) => s.currentDocumentId)
  const mode = useEditorUiStore((s) => s.mode)
  const isProcessingRef = useRef(false)

  const canUndo = useTextBlockHistoryStore((s) =>
    currentDocumentId ? s.canUndo(currentDocumentId) : false,
  )
  const canRedo = useTextBlockHistoryStore((s) =>
    currentDocumentId ? s.canRedo(currentDocumentId) : false,
  )

  const invalidate = async (docId: string) => {
    await queryClient.invalidateQueries({
      queryKey: getGetDocumentQueryKey(docId),
    })
    await queryClient.invalidateQueries({
      queryKey: getListDocumentsQueryKey(),
    })
  }

  const handleUndo = async () => {
    if (!currentDocumentId || isProcessingRef.current || !canUndo) return
    const history = useTextBlockHistoryStore.getState()
    const snapshot = history.popUndo(currentDocumentId)
    if (!snapshot) return

    // Push the current state onto the redo stack before restoring.
    const currentBlocks =
      currentDocument?.textBlocks?.map((b) => ({
        id: b.id ?? null,
        x: b.x,
        y: b.y,
        width: b.width,
        height: b.height,
        text: b.text ?? null,
        translation: b.translation ?? null,
        style: (b.style as any) ?? null,
      })) ?? []
    history.pushRedo(currentDocumentId, currentBlocks)

    isProcessingRef.current = true
    try {
      await putTextBlocks(currentDocumentId, snapshot)
      await invalidate(currentDocumentId)
      useEditorUiStore.getState().setSelectedBlockIndex(undefined)
    } catch (err) {
      // Re-push to undo so the user can retry.
      history.push(currentDocumentId, snapshot)
      history.popRedo(currentDocumentId)
      useEditorUiStore.getState().showError('Failed to undo text block operation')
      console.error('[textBlockUndo] undo failed:', err)
    } finally {
      isProcessingRef.current = false
    }
  }

  const handleRedo = async () => {
    if (!currentDocumentId || isProcessingRef.current || !canRedo) return
    const history = useTextBlockHistoryStore.getState()
    const snapshot = history.popRedo(currentDocumentId)
    if (!snapshot) return

    // Push the current state onto the undo stack before restoring.
    const currentBlocks =
      currentDocument?.textBlocks?.map((b) => ({
        id: b.id ?? null,
        x: b.x,
        y: b.y,
        width: b.width,
        height: b.height,
        text: b.text ?? null,
        translation: b.translation ?? null,
        style: (b.style as any) ?? null,
      })) ?? []
    history.push(currentDocumentId, currentBlocks)

    isProcessingRef.current = true
    try {
      await putTextBlocks(currentDocumentId, snapshot)
      await invalidate(currentDocumentId)
      useEditorUiStore.getState().setSelectedBlockIndex(undefined)
    } catch (err) {
      history.pushRedo(currentDocumentId, snapshot)
      history.popUndo(currentDocumentId)
      useEditorUiStore.getState().showError('Failed to redo text block operation')
      console.error('[textBlockUndo] redo failed:', err)
    } finally {
      isProcessingRef.current = false
    }
  }

  useEffect(() => {
    if (!enabled) return

    const handler = (e: KeyboardEvent) => {
      // Only active in select or block modes (not while brushing — brush has its own handler).
      if (mode !== 'select' && mode !== 'block') return

      const isMac =
        navigator.platform.includes('Mac') ||
        navigator.userAgent.includes('Mac')
      const modifier = isMac ? e.metaKey : e.ctrlKey

      if (!modifier || e.key !== 'z') return

      // Don't intercept when focus is in an input.
      const target = e.target as HTMLElement
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return
      }

      e.preventDefault()
      if (e.shiftKey) {
        void handleRedo()
      } else {
        void handleUndo()
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [enabled, mode, currentDocumentId, canUndo, canRedo])
}
