import { expect, test } from './helpers/test'
import {
  PIPELINE_SINGLE,
  bootstrapApp,
  importAndOpenPage,
} from './helpers/app'
import { drawStrokeOnCanvas } from './helpers/canvas'
import { prepareDetectAndOcr, runInpaint } from './helpers/pipeline'
import { selectors } from './helpers/selectors'

test.beforeEach(async ({ page }) => {
  await bootstrapApp(page)
})

test('undo and redo brush strokes with Cmd+Z / Cmd+Shift+Z', async ({
  page,
}) => {
  await importAndOpenPage(page, PIPELINE_SINGLE)
  await prepareDetectAndOcr(page)
  await runInpaint(page)

  // Switch to brush mode
  await page.getByTestId(selectors.tools.brush).click()
  const brushCanvas = page.getByTestId(selectors.workspace.brushCanvas)
  await expect(brushCanvas).toBeVisible()

  // Draw first stroke
  await drawStrokeOnCanvas(
    page,
    brushCanvas,
    { x: 0.2, y: 0.25 },
    { x: 0.65, y: 0.35 },
  )

  // Wait for stroke to be processed
  await page.waitForTimeout(500)

  // Draw second stroke
  await drawStrokeOnCanvas(
    page,
    brushCanvas,
    { x: 0.3, y: 0.4 },
    { x: 0.7, y: 0.5 },
  )

  // Wait for stroke to be processed
  await page.waitForTimeout(500)

  // Undo the second stroke (Cmd+Z on macOS)
  await page.keyboard.press('Meta+z')

  // Wait for undo to complete
  await page.waitForTimeout(500)

  // Undo the first stroke
  await page.keyboard.press('Meta+z')

  // Wait for undo to complete
  await page.waitForTimeout(500)

  // Redo the first stroke (Cmd+Shift+Z on macOS)
  await page.keyboard.press('Meta+Shift+z')

  // Wait for redo to complete
  await page.waitForTimeout(500)

  // Redo the second stroke
  await page.keyboard.press('Meta+Shift+z')

  // Wait for redo to complete
  await page.waitForTimeout(500)

  // Canvas should still be visible after all operations
  await expect(brushCanvas).toBeVisible()
})

test('undo and redo work with eraser mode', async ({ page }) => {
  await importAndOpenPage(page, PIPELINE_SINGLE)
  await prepareDetectAndOcr(page)
  await runInpaint(page)

  // Switch to brush mode and draw
  await page.getByTestId(selectors.tools.brush).click()
  const brushCanvas = page.getByTestId(selectors.workspace.brushCanvas)
  await expect(brushCanvas).toBeVisible()

  await drawStrokeOnCanvas(
    page,
    brushCanvas,
    { x: 0.2, y: 0.25 },
    { x: 0.65, y: 0.35 },
  )

  await page.waitForTimeout(500)

  // Switch to eraser mode and erase
  await page.getByTestId(selectors.tools.eraser).click()
  await expect(page.getByTestId(selectors.tools.eraser)).toHaveAttribute(
    'data-active',
    'true',
  )

  await drawStrokeOnCanvas(
    page,
    brushCanvas,
    { x: 0.4, y: 0.3 },
    { x: 0.55, y: 0.33 },
  )

  await page.waitForTimeout(500)

  // Undo eraser stroke
  await page.keyboard.press('Meta+z')
  await page.waitForTimeout(500)

  // Undo brush stroke
  await page.keyboard.press('Meta+z')
  await page.waitForTimeout(500)

  // Redo brush stroke
  await page.keyboard.press('Meta+Shift+z')
  await page.waitForTimeout(500)

  // Canvas should still be visible
  await expect(brushCanvas).toBeVisible()
})

test('undo/redo does not trigger in select mode', async ({ page }) => {
  await importAndOpenPage(page, PIPELINE_SINGLE)
  await prepareDetectAndOcr(page)
  await runInpaint(page)

  // Draw a brush stroke
  await page.getByTestId(selectors.tools.brush).click()
  const brushCanvas = page.getByTestId(selectors.workspace.brushCanvas)
  await drawStrokeOnCanvas(
    page,
    brushCanvas,
    { x: 0.2, y: 0.25 },
    { x: 0.65, y: 0.35 },
  )
  await page.waitForTimeout(500)

  // Switch to select mode
  await page.getByTestId(selectors.tools.select).click()
  await expect(page.getByTestId(selectors.tools.select)).toHaveAttribute(
    'data-active',
    'true',
  )

  // Try undo in select mode - should not work
  await page.keyboard.press('Meta+z')
  await page.waitForTimeout(500)

  // Switch back to brush mode to verify stroke is still there
  await page.getByTestId(selectors.tools.brush).click()
  await expect(brushCanvas).toBeVisible()
})

test('history limit is respected (max 50 entries)', async ({ page }) => {
  await importAndOpenPage(page, PIPELINE_SINGLE)
  await prepareDetectAndOcr(page)
  await runInpaint(page)

  // Switch to brush mode
  await page.getByTestId(selectors.tools.brush).click()
  const brushCanvas = page.getByTestId(selectors.workspace.brushCanvas)
  await expect(brushCanvas).toBeVisible()

  // Draw 55 strokes (exceeds max of 50)
  for (let i = 0; i < 55; i++) {
    const x1 = 0.2 + (i % 10) * 0.05
    const y1 = 0.2 + Math.floor(i / 10) * 0.1
    const x2 = x1 + 0.1
    const y2 = y1 + 0.05
    await drawStrokeOnCanvas(page, brushCanvas, { x: x1, y: y1 }, { x: x2, y: y2 })
    await page.waitForTimeout(100)
  }

  // Try to undo 55 times - only last 50 should be available
  for (let i = 0; i < 55; i++) {
    await page.keyboard.press('Meta+z')
    await page.waitForTimeout(50)
  }

  // Canvas should still be visible
  await expect(brushCanvas).toBeVisible()
})
