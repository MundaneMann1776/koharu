'use client'

import { useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import {
  useGetDocument,
  getGetDocumentQueryKey,
  getListDocumentsQueryKey,
} from '@/lib/api/documents/documents'
export { useBlobData, useDocumentLayer } from '@/hooks/useBlobData'
import {
  createTextBlock,
  patchTextBlock,
  putTextBlocks,
} from '@/lib/api/text-blocks/text-blocks'
import { useEditorUiStore } from '@/lib/stores/editorUiStore'
import { useTextBlockHistoryStore } from '@/lib/stores/textBlockHistoryStore'
import { TextBlock } from '@/types'
import type { DocumentDetail, TextBlockInput } from '@/lib/api/schemas'

const hasGeometryChange = (updates: Partial<TextBlock>) =>
  Object.prototype.hasOwnProperty.call(updates, 'x') ||
  Object.prototype.hasOwnProperty.call(updates, 'y') ||
  Object.prototype.hasOwnProperty.call(updates, 'width') ||
  Object.prototype.hasOwnProperty.call(updates, 'height')

const mapTextBlock = (
  block: DocumentDetail['textBlocks'][number],
): TextBlock => ({
  id: block.id,
  x: block.x,
  y: block.y,
  width: block.width,
  height: block.height,
  confidence: block.confidence,
  linePolygons: block.linePolygons as TextBlock['linePolygons'],
  sourceDirection: block.sourceDirection ?? undefined,
  renderedDirection: block.renderedDirection ?? undefined,
  sourceLanguage: block.sourceLanguage ?? undefined,
  rotationDeg: block.rotationDeg ?? undefined,
  detectedFontSizePx: block.detectedFontSizePx ?? undefined,
  detector: block.detector ?? undefined,
  text: block.text ?? undefined,
  translation: block.translation ?? undefined,
  style: block.style as TextBlock['style'],
  fontPrediction: block.fontPrediction as TextBlock['fontPrediction'],
  rendered: block.rendered ?? undefined,
  renderX: block.renderX ?? undefined,
  renderY: block.renderY ?? undefined,
  renderWidth: block.renderWidth ?? undefined,
  renderHeight: block.renderHeight ?? undefined,
})

export type MappedDocument = {
  id: string
  name: string
  width: number
  height: number
  textBlocks: TextBlock[]
  /** Blob hashes for each layer — fetch bytes via useDocumentLayer(). */
  image: string
  segment?: string
  inpainted?: string
  brushLayer?: string
  rendered?: string
  style?: { defaultFont?: string | null }
}

const mapDocumentDetail = (detail: DocumentDetail): MappedDocument => ({
  id: detail.id,
  name: detail.name,
  width: detail.width,
  height: detail.height,
  textBlocks: detail.textBlocks.map(mapTextBlock),
  image: detail.image,
  segment: detail.segment ?? undefined,
  inpainted: detail.inpainted ?? undefined,
  brushLayer: detail.brushLayer ?? undefined,
  rendered: detail.rendered ?? undefined,
  style: detail.style ?? undefined,
})

const toTextBlockInput = (block: TextBlock): TextBlockInput => ({
  id: block.id ?? null,
  x: block.x,
  y: block.y,
  width: block.width,
  height: block.height,
  text: block.text ?? null,
  translation: block.translation ?? null,
  style: (block.style as any) ?? null,
})

export function useCurrentDocument(): MappedDocument | null {
  const documentId = useEditorUiStore((s) => s.currentDocumentId)
  const { data: detail } = useGetDocument(documentId ?? '', {
    query: { enabled: !!documentId },
  })
  if (!detail) return null
  return mapDocumentDetail(detail)
}

export function useTextBlocks() {
  const queryClient = useQueryClient()
  const document = useCurrentDocument()
  const textBlocks = document?.textBlocks ?? []
  const selectedBlockIndex = useEditorUiStore(
    (state) => state.selectedBlockIndex,
  )
  const setSelectedBlockIndex = useEditorUiStore(
    (state) => state.setSelectedBlockIndex,
  )

  const invalidateDocument = useCallback(
    async (docId: string) => {
      await queryClient.invalidateQueries({
        queryKey: getGetDocumentQueryKey(docId),
      })
      await queryClient.invalidateQueries({
        queryKey: getListDocumentsQueryKey(),
      })
    },
    [queryClient],
  )

  const updateTextBlocks = useCallback(
    async (blocks: TextBlock[]) => {
      const docId = useEditorUiStore.getState().currentDocumentId
      if (!docId) return
      await putTextBlocks(docId, blocks.map(toTextBlockInput))
      await invalidateDocument(docId)
    },
    [invalidateDocument],
  )

  const replaceBlock = async (index: number, updates: Partial<TextBlock>) => {
    const docId = useEditorUiStore.getState().currentDocumentId
    if (!docId) return
    const block = document?.textBlocks?.[index]
    if (!block?.id) return

    const patch: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(updates)) {
      patch[key] = value
    }

    const geometryActuallyChanged =
      (updates.x !== undefined && updates.x !== block.x) ||
      (updates.y !== undefined && updates.y !== block.y) ||
      (updates.width !== undefined && updates.width !== block.width) ||
      (updates.height !== undefined && updates.height !== block.height)

    const queryKey = getGetDocumentQueryKey(docId)
    // Snapshot for rollback in case the patch request fails.
    const previousData = queryClient.getQueryData<DocumentDetail>(queryKey)

    if (geometryActuallyChanged) {
      const ui = useEditorUiStore.getState()
      ui.setShowRenderedImage(false)
      ui.setShowTextBlocksOverlay(true)

      // Optimistic cache update: apply new geometry while preserving style,
      // fontPrediction, detectedFontSizePx and all other fields so that the
      // RenderControlsPanel never flashes a different font size.
      if (previousData) {
        queryClient.setQueryData<DocumentDetail>(queryKey, {
          ...previousData,
          textBlocks: previousData.textBlocks.map((b) =>
            b.id === block.id
              ? {
                  ...b,
                  x: updates.x ?? b.x,
                  y: updates.y ?? b.y,
                  width: updates.width ?? b.width,
                  height: updates.height ?? b.height,
                }
              : b,
          ),
        })
      }
    }

    try {
      await patchTextBlock(docId, block.id, patch)
      await invalidateDocument(docId)
    } catch (error) {
      // Roll back the optimistic update so the cache doesn't stay stale.
      if (previousData) {
        queryClient.setQueryData<DocumentDetail>(queryKey, previousData)
      }
      throw error
    }
  }

  const appendBlock = async (block: TextBlock) => {
    const docId = useEditorUiStore.getState().currentDocumentId
    if (!docId) return
    // Snapshot before creating so Cmd+Z can delete the new block.
    const currentBlocks = document?.textBlocks ?? []
    useTextBlockHistoryStore
      .getState()
      .push(docId, currentBlocks.map(toTextBlockInput))
    await createTextBlock(docId, {
      x: block.x,
      y: block.y,
      width: block.width,
      height: block.height,
    })
    await invalidateDocument(docId)
    setSelectedBlockIndex(currentBlocks.length)
  }

  const removeBlock = async (index: number) => {
    const docId = useEditorUiStore.getState().currentDocumentId
    if (!docId) return
    const currentBlocks = document?.textBlocks ?? []
    // Snapshot before deleting so Cmd+Z can restore the block.
    useTextBlockHistoryStore
      .getState()
      .push(docId, currentBlocks.map(toTextBlockInput))
    const nextBlocks = currentBlocks.filter((_, idx) => idx !== index)
    await updateTextBlocks(nextBlocks)
    setSelectedBlockIndex(undefined)
  }

  const clearSelection = () => {
    setSelectedBlockIndex(undefined)
  }

  return {
    document,
    textBlocks,
    selectedBlockIndex,
    setSelectedBlockIndex,
    clearSelection,
    replaceBlock,
    appendBlock,
    removeBlock,
    updateTextBlocks,
  }
}
