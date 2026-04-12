'use client'

import { useRef, useLayoutEffect } from 'react'
import { useDrag } from '@use-gesture/react'
import { useHotkeys } from 'react-hotkeys-hook'
import { useEditorUiStore } from '@/lib/stores/editorUiStore'
import { TextBlock } from '@/types'
import { useTextBlocks } from '@/hooks/useTextBlocks'
import { useBlobImage } from '@/hooks/useBlobData'

type TextBlockLayerProps = {
  selectedIndex?: number
  onSelect: (index?: number) => void
  showSprites?: boolean
  scale: number
  style?: React.CSSProperties
}

export function TextBlockLayer({
  selectedIndex,
  onSelect,
  showSprites,
  scale,
  style,
}: TextBlockLayerProps) {
  const { textBlocks, replaceBlock, removeBlock } = useTextBlocks()
  const mode = useEditorUiStore((state) => state.mode)
  const interactive = mode === 'select' || mode === 'block'

  useHotkeys(
    'delete',
    () => {
      if (selectedIndex !== undefined && interactive) {
        void removeBlock(selectedIndex)
      }
    },
    { enabled: selectedIndex !== undefined && interactive },
    [selectedIndex, interactive, removeBlock],
  )

  return (
    <div
      data-text-block-layer
      style={{
        ...style,
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    >
      {textBlocks.map((block, index) => (
        <TextBlockItem
          key={block.id ?? index}
          block={block}
          index={index}
          scale={scale}
          showSprites={!!showSprites}
          selected={index === selectedIndex}
          interactive={interactive}
          onSelect={onSelect}
          onUpdate={(updates) => replaceBlock(index, updates)}
        />
      ))}
    </div>
  )
}

type TextBlockItemProps = {
  block: TextBlock
  index: number
  scale: number
  showSprites: boolean
  selected: boolean
  interactive: boolean
  onSelect: (index: number) => void
  onUpdate: (updates: Partial<TextBlock>) => Promise<void>
}

const RESIZE_HANDLE_SIZE = 8

type ResizeEdge = {
  top: boolean
  bottom: boolean
  left: boolean
  right: boolean
}

function TextBlockItem({
  block,
  index,
  scale,
  showSprites,
  selected,
  interactive,
  onSelect,
  onUpdate,
}: TextBlockItemProps) {
  const boxRef = useRef<HTMLDivElement>(null)
  // Store block values at drag start so we're immune to mid-drag re-renders.
  const dragStart = useRef({ x: 0, y: 0, w: 0, h: 0 })
  const edgeRef = useRef<ResizeEdge | null>(null)
  const isResizeRef = useRef(false)
  // Stays true from drag start until the server patch+refetch completes,
  // preventing useLayoutEffect from snapping the element back to the old
  // server position while the async update is in flight.
  const isDragging = useRef(false)

  // Sync block geometry to the DOM imperatively. This is the sole source of
  // truth for transform/width/height — they are intentionally NOT included in
  // the JSX style prop so React's reconciler never overwrites the values that
  // setBox writes during an active drag.
  useLayoutEffect(() => {
    if (isDragging.current) return
    const el = boxRef.current
    if (!el) return
    el.style.transition = 'transform 80ms ease-out'
    el.style.transform = `translate(${block.x * scale}px, ${block.y * scale}px)`
    el.style.width = `${block.width * scale}px`
    el.style.height = `${block.height * scale}px`
  }, [block.x, block.y, block.width, block.height, scale])

  const setBox = (x: number, y: number, w: number, h: number) => {
    const el = boxRef.current
    if (!el) return
    el.style.transition = 'none'
    el.style.transform = `translate(${x}px, ${y}px)`
    el.style.width = `${w}px`
    el.style.height = `${h}px`
  }

  const bind = useDrag(
    ({ first, last, movement: [mx, my], event, tap }) => {
      if (!interactive) return
      event?.stopPropagation()
      if (tap) {
        onSelect(index)
        return
      }

      if (first) {
        isDragging.current = true
        // Snapshot block state at drag start.
        dragStart.current = {
          x: block.x * scale,
          y: block.y * scale,
          w: block.width * scale,
          h: block.height * scale,
        }
        onSelect(index)
      }

      const { x: sx, y: sy, w: sw, h: sh } = dragStart.current
      const edge = edgeRef.current

      if (isResizeRef.current && edge) {
        let dx = 0
        let dy = 0
        let w = sw
        let h = sh

        if (edge.right) w += mx
        if (edge.left) {
          w -= mx
          dx = mx
        }
        if (edge.bottom) h += my
        if (edge.top) {
          h -= my
          dy = my
        }

        w = Math.max(4 * scale, w)
        h = Math.max(4 * scale, h)
        if (edge.left && w === 4 * scale) dx = sw - 4 * scale
        if (edge.top && h === 4 * scale) dy = sh - 4 * scale

        setBox(sx + dx, sy + dy, w, h)

        if (last) {
          isResizeRef.current = false
          edgeRef.current = null
          void onUpdate({
            x: Math.round((sx + dx) / scale),
            y: Math.round((sy + dy) / scale),
            width: Math.max(4, Math.round(w / scale)),
            height: Math.max(4, Math.round(h / scale)),
          }).finally(() => {
            isDragging.current = false
          })
        }
      } else {
        setBox(sx + mx, sy + my, sw, sh)

        if (last) {
          void onUpdate({
            x: Math.round((sx + mx) / scale),
            y: Math.round((sy + my) / scale),
          }).finally(() => {
            isDragging.current = false
          })
        }
      }
    },
    {
      pointer: { buttons: 1, touch: true },
      filterTaps: true,
      preventDefault: true,
      eventOptions: { passive: false },
    },
  )

  const handleEdgePointerDown = (edge: ResizeEdge) => {
    if (!interactive || !selected) return
    isResizeRef.current = true
    edgeRef.current = edge
  }

  const spriteOffsetX = ((block.renderX ?? block.x) - block.x) * scale
  const spriteOffsetY = ((block.renderY ?? block.y) - block.y) * scale

  return (
    <div
      ref={boxRef}
      {...bind()}
      data-testid={`workspace-text-block-${index}`}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        // transform, width, height are managed exclusively by useLayoutEffect
        // and setBox — omitting them here prevents React's reconciler from
        // overwriting imperative DOM updates during an active drag.
        pointerEvents: interactive ? 'auto' : 'none',
        zIndex: selected ? 20 : 10,
        touchAction: 'none',
        cursor: interactive ? 'move' : 'default',
        willChange: 'transform',
      }}
    >
      {/* Always mounted so the blob is prefetched while the composite rendered
          image is visible. Visibility is toggled rather than unmounting so the
          blob URL stays cached and appears instantly when showSprites flips. */}
      <BlockSprite
        block={block}
        scale={scale}
        x={spriteOffsetX}
        y={spriteOffsetY}
        index={index}
        visible={showSprites}
      />

      {/* Annotation border */}
      <div
        className={`absolute inset-0 rounded ${
          selected
            ? 'border-primary bg-primary/15 border-[3px]'
            : 'border-2 border-rose-400/60 bg-rose-400/5'
        }`}
      />

      {/* Index badge */}
      <div
        className={`pointer-events-none absolute -top-1.5 -left-1.5 flex h-4 w-4 items-center justify-center rounded-full text-[9px] font-semibold text-white shadow ${
          selected ? 'bg-primary' : 'bg-rose-400'
        }`}
      >
        {index + 1}
      </div>

      {/* Resize handles */}
      {selected && interactive && (
        <ResizeHandles onEdgePointerDown={handleEdgePointerDown} />
      )}
    </div>
  )
}

function BlockSprite({
  block,
  scale,
  x,
  y,
  index,
  visible,
}: {
  block: TextBlock
  scale: number
  x: number
  y: number
  index: number
  visible: boolean
}) {
  const { data: src } = useBlobImage(block.rendered)
  if (!src) return null
  return (
    <img
      alt=''
      src={src}
      draggable={false}
      data-testid={`workspace-text-block-sprite-${index}`}
      className='pointer-events-none absolute select-none'
      style={{
        top: 0,
        left: 0,
        transformOrigin: 'top left',
        transform: `translate(${x}px, ${y}px) scale(${scale})`,
        visibility: visible ? 'visible' : 'hidden',
      }}
    />
  )
}

function ResizeHandles({
  onEdgePointerDown,
}: {
  onEdgePointerDown: (edge: ResizeEdge) => void
}) {
  const s = RESIZE_HANDLE_SIZE
  const half = s / 2

  const edges: {
    edge: ResizeEdge
    style: React.CSSProperties
    cursor: string
  }[] = [
    // Corners
    {
      edge: { top: true, left: true, bottom: false, right: false },
      cursor: 'nwse-resize',
      style: { top: -half, left: -half, width: s, height: s },
    },
    {
      edge: { top: true, left: false, bottom: false, right: true },
      cursor: 'nesw-resize',
      style: { top: -half, right: -half, width: s, height: s },
    },
    {
      edge: { top: false, left: true, bottom: true, right: false },
      cursor: 'nesw-resize',
      style: { bottom: -half, left: -half, width: s, height: s },
    },
    {
      edge: { top: false, left: false, bottom: true, right: true },
      cursor: 'nwse-resize',
      style: { bottom: -half, right: -half, width: s, height: s },
    },
    // Edges
    {
      edge: { top: true, left: false, bottom: false, right: false },
      cursor: 'ns-resize',
      style: { top: -half, left: s, right: s, height: s },
    },
    {
      edge: { top: false, left: false, bottom: true, right: false },
      cursor: 'ns-resize',
      style: { bottom: -half, left: s, right: s, height: s },
    },
    {
      edge: { top: false, left: true, bottom: false, right: false },
      cursor: 'ew-resize',
      style: { left: -half, top: s, bottom: s, width: s },
    },
    {
      edge: { top: false, left: false, bottom: false, right: true },
      cursor: 'ew-resize',
      style: { right: -half, top: s, bottom: s, width: s },
    },
  ]

  return (
    <>
      {edges.map((e, i) => (
        <div
          key={i}
          onPointerDown={() => onEdgePointerDown(e.edge)}
          style={{
            position: 'absolute',
            ...e.style,
            cursor: e.cursor,
            zIndex: 30,
          }}
        />
      ))}
    </>
  )
}
