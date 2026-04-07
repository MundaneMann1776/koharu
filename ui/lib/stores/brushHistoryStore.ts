'use client'

import { create } from 'zustand'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type HistoryEntry = {
  documentId: string
  brushLayerPng: Uint8Array // Full brush layer snapshot
  timestamp: number
}

type BrushHistoryState = {
  // Per-document history stacks (max 50 entries per document)
  undoStacks: Map<string, HistoryEntry[]>
  redoStacks: Map<string, HistoryEntry[]>

  // Actions
  pushHistory: (docId: string, layerPng: Uint8Array) => void
  undo: (docId: string) => HistoryEntry | null
  redo: (docId: string) => HistoryEntry | null
  clear: (docId: string) => void
  clearAll: () => void
  canUndo: (docId: string) => boolean
  canRedo: (docId: string) => boolean
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_HISTORY_ENTRIES = 50

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useBrushHistoryStore = create<BrushHistoryState>((set, get) => ({
  undoStacks: new Map(),
  redoStacks: new Map(),

  pushHistory: (docId, layerPng) => {
    set((state) => {
      const undoStacks = new Map(state.undoStacks)
      const redoStacks = new Map(state.redoStacks)

      const undoStack = undoStacks.get(docId) || []
      const newEntry: HistoryEntry = {
        documentId: docId,
        brushLayerPng: layerPng,
        timestamp: Date.now(),
      }

      // Add to undo stack
      const updatedUndoStack = [...undoStack, newEntry]

      // Trim to max entries (remove oldest)
      if (updatedUndoStack.length > MAX_HISTORY_ENTRIES) {
        updatedUndoStack.shift()
      }

      undoStacks.set(docId, updatedUndoStack)

      // Clear redo stack on new operation (standard undo/redo behavior)
      redoStacks.set(docId, [])

      return { undoStacks, redoStacks }
    })
  },

  undo: (docId) => {
    const state = get()
    const undoStack = state.undoStacks.get(docId) || []
    const redoStack = state.redoStacks.get(docId) || []

    if (undoStack.length === 0) return null

    const undoStacks = new Map(state.undoStacks)
    const redoStacks = new Map(state.redoStacks)

    // Pop from undo stack
    const newUndoStack = [...undoStack]
    const entry = newUndoStack.pop()!

    // Push current state to redo stack (will be set by caller)
    undoStacks.set(docId, newUndoStack)

    set({ undoStacks, redoStacks })
    return entry
  },

  redo: (docId) => {
    const state = get()
    const redoStack = state.redoStacks.get(docId) || []

    if (redoStack.length === 0) return null

    const redoStacks = new Map(state.redoStacks)

    // Pop from redo stack
    const newRedoStack = [...redoStack]
    const entry = newRedoStack.pop()!

    redoStacks.set(docId, newRedoStack)

    set({ redoStacks })
    return entry
  },

  clear: (docId) => {
    set((state) => {
      const undoStacks = new Map(state.undoStacks)
      const redoStacks = new Map(state.redoStacks)

      undoStacks.delete(docId)
      redoStacks.delete(docId)

      return { undoStacks, redoStacks }
    })
  },

  clearAll: () => {
    set({ undoStacks: new Map(), redoStacks: new Map() })
  },

  canUndo: (docId) => {
    const undoStack = get().undoStacks.get(docId) || []
    return undoStack.length > 0
  },

  canRedo: (docId) => {
    const redoStack = get().redoStacks.get(docId) || []
    return redoStack.length > 0
  },
}))
